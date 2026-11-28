"""Phase 5 entry point. Run from the repository root."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from sarchaeology.candidates.pipeline import build,verify_products
from sarchaeology.candidates.products import transform_private_package
from sarchaeology.candidates.review import feedback
from sarchaeology.candidates.validation import evaluate_independent
from sarchaeology.fusion.io import read_json,save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["build","verify","validate","feedback","seal","unseal"])
    parser.add_argument("--config",type=Path,default=Path("configs/phase5/buto.yaml"))
    parser.add_argument("--run",type=Path,default=Path("outputs/phase5/buto_2024"))
    parser.add_argument("--out",type=Path)
    parser.add_argument("--study",action="append",type=Path,default=[])
    parser.add_argument("--k",type=int,default=10)
    parser.add_argument("--key-file",type=Path)
    parser.add_argument("--encrypted",type=Path,help="Encrypted input package for unseal")
    args = parser.parse_args()
    if args.action=="verify":
        verify_products(args.run)
        return
    target = args.run if args.action=="build" else args.out
    if target is None or target.exists():
        parser.error("Choose a fresh destination (--run for build, --out for validation or feedback)")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        out = Path(tmp)/"result"
        if args.action=="build":
            build(args.config,out)
            verify_products(out)
        elif args.action in ("seal","unseal"):
            if not args.key_file or (args.action=="unseal" and not args.encrypted):
                parser.error("Provide --key-file and, for unseal, --encrypted")
            source = args.encrypted if args.action=="unseal" else args.run/"restricted_research.zip"
            transform_private_package(source,args.key_file,out,decrypt=args.action=="unseal")
        else:
            p = read_json(args.run/"provenance.json")
            if args.action=="validate":
                evaluate_independent(args.study,Path(p["config"]["prepared"]),p["policy_sha256"],p["model_sha256"],out,args.k)
            else:
                out.mkdir()
                hist,latest,agreement = feedback(args.run/"feedback/expert.sqlite",p["catalogue_sha256"])
                hist.to_csv(out/"review_history.csv",index=False)
                latest.to_csv(out/"latest_reviews.csv",index=False)
                save_json(out/"reviewer_agreement.json",agreement)
        out.rename(target)


if __name__=="__main__":
    main()
