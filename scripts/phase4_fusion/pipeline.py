"""Phase 4 command line. Run from the repository root."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from sarchaeology.fusion.model import VARIANTS


@contextmanager
def publish_directory(destination):
    """Expose a complete stage only after all files have been written successfully."""
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-",dir=destination.parent) as staging:
        result = Path(staging)/"result"
        yield result
        result.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    for name in ("prepare","train","export","evaluate","verify","align-heldout"):
        cmd = sub.add_parser(name)
        if name in ("prepare","train","export"):
            cmd.add_argument("--config",type=Path,default=Path("configs/phase4/buto_2024.yaml"))
        cmd.add_argument("--prepared",type=Path,default=Path("outputs/phase4/buto_2024"))
        if name in ("train","evaluate"):
            cmd.add_argument("--runs",type=Path,default=Path("outputs/phase4/buto_runs"))
        if name in ("export","evaluate","verify"):
            cmd.add_argument("--maps",type=Path,default=Path("outputs/phase4/buto_maps"))
        if name == "train":
            cmd.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS))
            cmd.add_argument("--epochs",type=int)
            cmd.add_argument("--alignment-weight",type=float,help="Set to zero for the alignment ablation")
            cmd.add_argument("--temporal-weight",type=float,help="Set to zero for the temporal ablation")
        if name in ("export","align-heldout"):
            cmd.add_argument("--model",type=Path,default=Path("outputs/phase4/buto_runs/gated"))
        if name == "align-heldout":
            cmd.add_argument("--destination",type=Path,default=Path("outputs/phase4/buto_heldout"))
        if name == "evaluate":
            cmd.add_argument("--report",type=Path,default=Path("outputs/phase4/buto_evaluation"))
            cmd.add_argument("--labels",type=Path)
            cmd.add_argument("--comparison",type=Path,action="append",default=[],
                             help="Include an extra trained model directory from a controlled ablation")
        if name == "verify":
            cmd.add_argument("--package",type=Path,help="Write the verified maps and metadata to a new ZIP")
            cmd.add_argument("--heldout",type=Path,help="Also verify the separate inference-only cache")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text()) if hasattr(args,"config") else None
    if args.action == "prepare":
        from sarchaeology.fusion.prepare import prepare
        with publish_directory(args.prepared) as destination:
            prepare(cfg,destination)
    elif args.action == "align-heldout":
        from sarchaeology.fusion.prepare import prepare
        with publish_directory(args.destination) as destination:
            prepare(None,destination,frozen_data=args.prepared,model_dir=args.model)
    elif args.action == "train":
        from sarchaeology.fusion.training import train_all
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        for key in ("alignment_weight","temporal_weight"):
            value = getattr(args,key)
            if value is not None:
                if value < 0:
                    parser.error(f"{key} must be nonnegative")
                cfg[key] = value
        with publish_directory(args.runs) as destination:
            train_all(args.prepared,destination,cfg,args.variants)
    elif args.action == "export":
        from sarchaeology.fusion.export import export
        with publish_directory(args.maps) as destination:
            export(args.prepared,args.model,destination,cfg)
    elif args.action == "evaluate":
        from sarchaeology.fusion.evaluation import evaluate
        with publish_directory(args.report) as destination:
            evaluate(args.runs,args.maps,destination,args.prepared,args.labels,args.comparison)
    elif args.action == "verify":
        from sarchaeology.fusion.evaluation import verify, verify_heldout
        verify(args.prepared,args.maps)
        if args.heldout:
            verify_heldout(args.prepared,args.heldout)
        if args.package:
            if args.package.exists():
                raise FileExistsError(f"Choose a fresh package name: {args.package}")
            args.package.parent.mkdir(parents=True,exist_ok=True)
            with TemporaryDirectory(dir=args.package.parent) as temp:
                archive = Path(temp)/"maps.zip"
                with ZipFile(archive,"x",ZIP_DEFLATED) as bundle:
                    for path in sorted(args.maps.iterdir()):
                        if path.is_file():
                            bundle.write(path,f"maps/{path.name}")
                    for name in ("terrain_statistics.json","alignment_qa.json"):
                        bundle.write(args.prepared/name,f"metadata/{name}")
                archive.rename(args.package)
            print(f"Packaged verified layers in {args.package}")


if __name__=="__main__":
    main()
