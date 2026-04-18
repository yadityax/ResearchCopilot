"""
Bulk ingestion of foundational AI/ML/CV papers into ResearchCopilot.
Covers foundational DL, NLP, biometrics, deepfakes, adversarial ML, RAG —
aligned with Prof. Mayank Vatsa's research areas and the project's RAG stack.

Usage:
    python scripts/ingest_foundational_papers.py
    python scripts/ingest_foundational_papers.py --dry-run
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from backend.config import get_settings
from backend.services.ingest_pipeline import IngestPipeline

# ── Paper list ────────────────────────────────────────────────────────────────
# Each entry: (search_query, max_results, description)
# max_results=1 for precise title searches, 2-3 for broader topic coverage.

PAPERS = [
    # ── Transformers & Attention ─────────────────────────────────────────────
    ("Attention Is All You Need Vaswani 2017 transformer", 1, "Transformers / Self-Attention"),
    ("BERT pre-training deep bidirectional transformers Devlin 2018", 1, "BERT"),
    ("GPT-3 language models few-shot learners Brown 2020", 1, "GPT-3"),
    ("RoBERTa robustly optimized BERT pretraining Liu 2019", 1, "RoBERTa"),

    # ── Foundational Deep Learning ───────────────────────────────────────────
    ("Deep Residual Learning Image Recognition He 2015 ResNet", 1, "ResNet"),
    ("ImageNet classification deep convolutional neural networks Krizhevsky AlexNet", 1, "AlexNet"),
    ("Dropout simple way prevent neural networks overfitting Srivastava", 1, "Dropout"),
    ("Batch Normalization accelerating deep networks Ioffe Szegedy 2015", 1, "Batch Norm"),

    # ── Word Embeddings & NLP ────────────────────────────────────────────────
    ("Efficient estimation word representations vector space Mikolov word2vec 2013", 1, "Word2Vec"),
    ("GloVe global vectors word representation Pennington 2014", 1, "GloVe"),
    ("ELMo deep contextualized word representations Peters 2018", 1, "ELMo"),
    ("Sentence-BERT sentence embeddings siamese BERT Reimers 2019", 1, "Sentence-BERT"),

    # ── Generative Models ────────────────────────────────────────────────────
    ("Generative Adversarial Nets Goodfellow 2014", 1, "GAN"),
    ("Auto-encoding variational bayes Kingma Welling VAE 2013", 1, "VAE"),
    ("Denoising diffusion probabilistic models Ho 2020 DDPM", 1, "DDPM"),
    ("High-resolution image synthesis latent diffusion models Rombach stable diffusion", 1, "Latent Diffusion / Stable Diffusion"),

    # ── RAG & Retrieval ──────────────────────────────────────────────────────
    ("Retrieval-augmented generation knowledge-intensive NLP Lewis 2020 RAG", 1, "RAG"),
    ("Dense passage retrieval open-domain question answering Karpukhin DPR 2020", 1, "DPR"),
    ("FAISS billion-scale similarity search GPU Johnson 2017", 1, "FAISS"),

    # ── Reinforcement Learning ───────────────────────────────────────────────
    ("Playing Atari deep reinforcement learning Mnih DQN 2013", 1, "DQN"),
    ("Proximal policy optimization algorithms Schulman PPO 2017", 1, "PPO"),

    # ── Face Recognition & Biometrics (Prof. Vatsa's core area) ─────────────
    ("FaceNet unified embedding face recognition clustering Schroff 2015", 1, "FaceNet"),
    ("ArcFace additive angular margin loss deep face recognition Deng 2018", 1, "ArcFace"),
    ("DeepFace closing gap human level performance face verification Taigman 2014", 1, "DeepFace"),
    ("CosFace large margin cosine loss deep face recognition Wang 2018", 1, "CosFace"),
    ("Face recognition survey deep learning Wang Deng 2021", 1, "Face Recognition Survey"),

    # ── Deepfake & Forgery Detection (Prof. Vatsa's active research) ────────
    ("FaceForensics++ learning detect manipulated facial images Rossler 2019", 1, "FaceForensics++"),
    ("MesoNet compact facial video forgery detection Afchar 2018", 1, "MesoNet"),
    ("Detecting deepfake videos artifacts face warping Li 2018", 1, "Deepfake Detection Artifacts"),
    ("Deepfake detection survey challenge Tolosana 2020", 1, "Deepfake Detection Survey"),

    # ── Adversarial Robustness (Prof. Vatsa publishes here) ─────────────────
    ("Explaining harnessing adversarial examples Goodfellow FGSM 2014", 1, "FGSM / Adversarial Examples"),
    ("Towards deep learning models resistant adversarial attacks Madry PGD 2017", 1, "PGD Attack"),
    ("Certified adversarial robustness randomized smoothing Cohen 2019", 1, "Certified Robustness"),
    ("Adversarial examples transferability physical world Kurakin 2016", 1, "Adversarial Transferability"),

    # ── Computer Vision Foundations (Prof. Vatsa teaches CV) ────────────────
    ("Very deep convolutional networks large-scale image recognition VGG Simonyan 2014", 1, "VGGNet"),
    ("Going deeper with convolutions GoogLeNet Inception Szegedy 2014", 1, "GoogLeNet / Inception"),
    ("Feature pyramid networks object detection Lin FPN 2017", 1, "FPN"),
    ("You only look once unified real-time object detection YOLO Redmon 2015", 1, "YOLO"),

    # ── Privacy & Federated Learning (his privacy research) ─────────────────
    ("Deep learning differential privacy Abadi 2016", 1, "DP-SGD"),
    ("Communication-efficient learning deep networks federated McMahan 2017", 1, "Federated Learning"),

    # ── MLOps & System (relevant to this project) ────────────────────────────
    ("MLflow managing machine learning lifecycle Zaharia 2018", 1, "MLflow"),
    ("Hidden technical debt machine learning systems Sculley 2015", 1, "ML Technical Debt"),
]

# ── Runner ────────────────────────────────────────────────────────────────────

async def run(dry_run: bool = False):
    settings = get_settings()
    pipeline = IngestPipeline(settings)

    total = len(PAPERS)
    success_count = 0
    fail_count = 0
    skip_count = 0

    print(f"\n{'='*65}")
    print(f"  ResearchCopilot — Foundational Paper Bulk Ingestion")
    print(f"  Total queries: {total}  |  dry_run={dry_run}")
    print(f"{'='*65}\n")

    for i, (query, max_results, description) in enumerate(PAPERS, 1):
        print(f"[{i:02d}/{total}] {description}")
        print(f"         Query: {query[:70]}...")

        if dry_run:
            print("         [DRY RUN — skipped]\n")
            skip_count += 1
            continue

        try:
            results = await pipeline.search_and_ingest(
                query=query,
                max_papers=max_results,
            )
            ok = [r for r in results if r.status == "success"]
            already = [r for r in results if r.status == "exists"]
            err = [r for r in results if r.status == "error"]

            for r in results:
                status_icon = "✓" if r.status == "success" else ("~" if r.status == "exists" else "✗")
                print(f"         {status_icon} [{r.status}] {r.paper_id} — chunks:{r.chunks_created} embeds:{r.embeddings_stored}")

            success_count += len(ok) + len(already)
            fail_count += len(err)

        except Exception as e:
            logger.error(f"Pipeline error for '{description}': {e}")
            print(f"         ✗ [ERROR] {e}")
            fail_count += 1

        print()
        # Small delay to avoid hammering arXiv API
        await asyncio.sleep(1.5)

    print(f"{'='*65}")
    print(f"  Done.  Ingested/existing: {success_count}  |  Failed: {fail_count}  |  Skipped: {skip_count}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest foundational AI papers into ResearchCopilot")
    parser.add_argument("--dry-run", action="store_true", help="List papers without ingesting")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
