from __future__ import annotations
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
KAGGLE_DATASET = 'wordsforthewise/lending-club'
OUT = RAW / 'lending_club_bundle.zip'


def download_via_kaggle_api() -> Path | None:
    try:
        api = KaggleApi()
        api.authenticate()
        print('[Kaggle] downloading...')
        api.dataset_download_files(KAGGLE_DATASET, path=str(RAW), unzip=False, quiet=False)
        candidates = [p for p in RAW.glob('*.zip') if p.stat().st_size > 100_000]
        if not candidates:
            return None
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        if newest != OUT:
            if OUT.exists():
                OUT.unlink()
            newest.replace(OUT)
        return OUT
    except Exception as exc:
        print(f'[Kaggle] download failed: {exc}')
        return None


def extract_accepted_only(bundle: Path) -> list[Path]:
    """Flat-extract only accepted CSVs; avoids the nested .csv folder issue."""
    import zipfile, shutil
    out = []
    with zipfile.ZipFile(bundle) as z:
        members = []
        for name in z.namelist():
            if name.lower().replace('\\', '/').endswith('.csv') and Path(name).name.lower().startswith('accepted_'):
                members.append(name)
        if not members:
            raise FileNotFoundError('Không tìm thấy accepted_*.csv trong Kaggle archive.')
        # Prefer the largest accepted member if the archive contains more than one.
        members.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        member = members[0]
        target = RAW / Path(member).name
        with z.open(member, 'r') as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        out.append(target)
    return out


if __name__ == '__main__':
    bundle = OUT if OUT.exists() else download_via_kaggle_api()
    if bundle is None:
        print('\nKhông tải được dataset. Hãy đăng nhập Kaggle bằng `kaggle auth login` rồi chạy lại.')
        sys.exit(2)
    files = extract_accepted_only(bundle)
    print('[Extract] accepted CSV:', [str(p.relative_to(ROOT)) for p in files])
    print('\nHoàn tất. Không cần dùng rejected CSV cho pipeline này.')
