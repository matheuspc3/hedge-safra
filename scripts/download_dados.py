"""Baixa dados climáticos da NASA POWER para as regiões de soja."""

import json
import socket

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# Forçar IPv4 — NASA POWER trava em IPv6 (Python 3.14+ tenta IPv6 primeiro)
import urllib3.util.connection as _uc
_uc.allowed_gai_family = lambda: socket.AF_INET

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
for d in [DATA_RAW, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

REGIOES = {
    "Sorriso_MT":    {"lat": -12.55, "lon": -55.71},
    "Londrina_PR":   {"lat": -23.31, "lon": -51.16},
    "Rio Verde_GO":  {"lat": -17.80, "lon": -50.93},
}

PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR"
START, END = "19810101", "20241231"


def baixar(lat, lon, nome):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={PARAMS}&community=AG"
        f"&longitude={lon:.4f}&latitude={lat:.4f}"
        f"&start={START}&end={END}&format=JSON"
    )
    with requests.get(url, timeout=90, stream=True, headers={"Accept": "application/json"}) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tqdm(
            desc=f"  {nome}", total=total, unit="B", unit_scale=True,
            leave=False, colour="green",
        ) as pb:
            chunks = []
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    chunks.append(chunk)
                    pb.update(len(chunk))
            dados = b"".join(chunks)
    parsed = json.loads(dados)["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in parsed.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"
    df.columns = [c.lower() for c in df.columns]
    df["regiao"] = nome
    print(f"  ✓ {nome}: {len(df)} dias  ({df.index.min().date()} — {df.index.max().date()})")
    return df


def processar(df):
    cols_temp = ["t2m", "t2m_max", "t2m_min"]
    for regiao in df["regiao"].unique():
        mask = df["regiao"] == regiao
        df.loc[mask, cols_temp] = df.loc[mask, cols_temp].interpolate(method="linear", limit=7)
    df.dropna(subset=cols_temp, inplace=True)
    df["t2m_amplitude"] = df["t2m_max"] - df["t2m_min"]
    df["ano"] = df.index.year
    df["mes"] = df.index.month
    df["safra"] = np.where(df.index.month >= 10, df.index.year, df.index.year - 1)
    return df


def main():
    cache = DATA_RAW / "nasa_power_raw.parquet"
    if cache.exists():
        print("Cache local encontrado. Carregando...")
        df = pd.read_parquet(cache)
        print(f"  {len(df)} registros de {df['regiao'].nunique()} regiões")
    else:
        print("Baixando dados da NASA POWER...")
        dfs = []
        for nome, coords in tqdm(list(REGIOES.items()), desc="Regiões", unit="região"):
            dfs.append(baixar(coords["lat"], coords["lon"], nome))
        df = pd.concat(dfs)
        df.to_parquet(cache)
        print(f"Cache salvo em {cache}")

    df = processar(df)
    out = DATA_PROCESSED / "temperatura_diaria_MT_PR_GO.parquet"
    df.to_parquet(out)
    print(f"Processado salvo em {out}")
    print(f"Total: {len(df)} registros, {df['regiao'].nunique()} regiões")


if __name__ == "__main__":
    main()
