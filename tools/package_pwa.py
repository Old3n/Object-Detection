"""Build a deployable source archive; inference requires the included backend."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / 'dist' / 'AutoPilot-pwa.zip'
    destination.parent.mkdir(exist_ok=True)
    files = [root / name for name in ('AutoPilot.py', 'requirements.txt', 'README.md')]
    files += sorted((root / 'web').glob('*'))
    files += [root / 'docs' / name for name in ('PORTING.md', 'RESEARCH_GUIDE.md')]
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    print(destination)


if __name__ == '__main__':
    main()
