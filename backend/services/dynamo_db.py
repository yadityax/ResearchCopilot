"""
DynamoDB Service — metadata store for papers and user sessions.
Uses local DynamoDB (Docker) in dev; AWS DynamoDB in production.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from loguru import logger

from backend.config import Settings


class DynamoDBService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._resource = None
        self._papers_table = None
        self._sessions_table = None
        self._user_sessions_table = None

    # ── Lazy init ──────────────────────────────────────────────────────────

    def _get_resource(self):
        if self._resource is None:
            self._resource = boto3.resource(
                "dynamodb",
                region_name=self.settings.aws_region,
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                endpoint_url=self.settings.dynamodb_endpoint_url,
            )
        return self._resource

    def _get_papers_table(self):
        if self._papers_table is None:
            self._papers_table = self._ensure_table(
                table_name=self.settings.dynamodb_papers_table,
                key_schema=[{"AttributeName": "paper_id", "KeyType": "HASH"}],
                attribute_definitions=[{"AttributeName": "paper_id", "AttributeType": "S"}],
            )
        return self._papers_table

    def _get_sessions_table(self):
        if self._sessions_table is None:
            self._sessions_table = self._ensure_table(
                table_name=self.settings.dynamodb_sessions_table,
                key_schema=[
                    {"AttributeName": "session_id", "KeyType": "HASH"},
                    {"AttributeName": "entry_id", "KeyType": "RANGE"},
                ],
                attribute_definitions=[
                    {"AttributeName": "session_id", "AttributeType": "S"},
                    {"AttributeName": "entry_id", "AttributeType": "S"},
                ],
            )
        return self._sessions_table

    def _get_user_sessions_table(self):
        if self._user_sessions_table is None:
            self._user_sessions_table = self._ensure_table(
                table_name="user_sessions",
                key_schema=[
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "session_id", "KeyType": "RANGE"},
                ],
                attribute_definitions=[
                    {"AttributeName": "user_id", "AttributeType": "S"},
                    {"AttributeName": "session_id", "AttributeType": "S"},
                ],
            )
        return self._user_sessions_table

    def _ensure_table(self, table_name, key_schema, attribute_definitions):
        resource = self._get_resource()
        try:
            table = resource.Table(table_name)
            table.load()
            logger.debug(f"[DynamoDB] Table '{table_name}' exists")
            return table
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"[DynamoDB] Creating table '{table_name}'")
                table = resource.create_table(
                    TableName=table_name,
                    KeySchema=key_schema,
                    AttributeDefinitions=attribute_definitions,
                    BillingMode="PAY_PER_REQUEST",
                )
                table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
                return table
            raise

    # ── Papers ─────────────────────────────────────────────────────────────

    async def put_paper(self, paper_id: str, title: str, chunk_count: int, **kwargs):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._put_paper_sync, paper_id, title, chunk_count, kwargs
        )

    def _put_paper_sync(self, paper_id, title, chunk_count, extra):
        table = self._get_papers_table()
        item = {
            "paper_id": paper_id,
            "title": title,
            "chunk_count": chunk_count,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        table.put_item(Item=item)
        logger.debug(f"[DynamoDB] stored paper metadata: {paper_id}")

    async def get_paper(self, paper_id: str) -> Optional[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_paper_sync, paper_id)

    def _get_paper_sync(self, paper_id: str) -> Optional[dict]:
        table = self._get_papers_table()
        response = table.get_item(Key={"paper_id": paper_id})
        return response.get("Item")

    async def scan_papers_by_keyword(self, keyword: str) -> List[dict]:
        """Scan all papers and return those whose title contains the keyword (case-insensitive)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._scan_papers_sync, keyword.lower())

    def _scan_papers_sync(self, keyword_lower: str) -> List[dict]:
        table = self._get_papers_table()
        try:
            response = table.scan(ProjectionExpression="paper_id, title")
            items = response.get("Items", [])
            matched = [
                item for item in items
                if keyword_lower in item.get("title", "").lower()
            ]
            logger.debug(f"[DynamoDB] scan_papers_by_keyword '{keyword_lower}' → {len(matched)} matches")
            return matched
        except Exception as e:
            logger.warning(f"[DynamoDB] scan failed: {e}")
            return []

    # ── Sessions / Memory ──────────────────────────────────────────────────

    async def put_memory_entry(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict,
    ) -> str:
        entry_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._put_entry_sync, session_id, entry_id, role, content, metadata
        )
        return entry_id

    def _put_entry_sync(self, session_id, entry_id, role, content, metadata):
        table = self._get_sessions_table()
        table.put_item(Item={
            "session_id": session_id,
            "entry_id": entry_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        })

    async def get_session_entries(self, session_id: str, limit: int = 10) -> List[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_entries_sync, session_id, limit)

    def _get_entries_sync(self, session_id: str, limit: int) -> List[dict]:
        table = self._get_sessions_table()
        response = table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            Limit=limit,
            ScanIndexForward=False,  # newest first
        )
        entries = response.get("Items", [])
        return list(reversed(entries))  # return in chronological order

    # ── User Chat Sessions ─────────────────────────────────────────────────

    async def create_user_session(self, user_id: str, session_id: str, session_name: str = "New Chat"):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._create_user_session_sync, user_id, session_id, session_name)

    def _create_user_session_sync(self, user_id: str, session_id: str, session_name: str):
        table = self._get_user_sessions_table()
        now = datetime.now(timezone.utc).isoformat()
        table.put_item(Item={
            "user_id": user_id,
            "session_id": session_id,
            "session_name": session_name,
            "created_at": now,
            "last_message_at": now,
            "message_count": 0,
        })

    async def get_user_sessions(self, user_id: str) -> List[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_user_sessions_sync, user_id)

    def _get_user_sessions_sync(self, user_id: str) -> List[dict]:
        table = self._get_user_sessions_table()
        response = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
        )
        return response.get("Items", [])

    async def rename_user_session(self, user_id: str, session_id: str, session_name: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._rename_session_sync, user_id, session_id, session_name)

    def _rename_session_sync(self, user_id: str, session_id: str, session_name: str):
        table = self._get_user_sessions_table()
        table.update_item(
            Key={"user_id": user_id, "session_id": session_id},
            UpdateExpression="SET session_name = :n",
            ExpressionAttributeValues={":n": session_name},
        )

    async def update_session_activity(self, user_id: str, session_id: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_activity_sync, user_id, session_id)

    def _update_activity_sync(self, user_id: str, session_id: str):
        table = self._get_user_sessions_table()
        now = datetime.now(timezone.utc).isoformat()
        table.update_item(
            Key={"user_id": user_id, "session_id": session_id},
            UpdateExpression="SET last_message_at = :t ADD message_count :one",
            ExpressionAttributeValues={":t": now, ":one": 1},
        )

    async def delete_user_session(self, user_id: str, session_id: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._delete_session_sync, user_id, session_id)

    def _delete_session_sync(self, user_id: str, session_id: str):
        table = self._get_user_sessions_table()
        table.delete_item(Key={"user_id": user_id, "session_id": session_id})
