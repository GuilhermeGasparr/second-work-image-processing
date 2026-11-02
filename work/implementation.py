"""
Atividade 2 - Processamento de Imagens (Prof. Matheus Araújo)

Questão 1: filtragem espacial (convolução 2D implementada manualmente) com h1..h11.
Questão 2: filtragem no domínio da frequência (FFT) e compressão por limiar.

Regras respeitadas:
  - PIL é usado APENAS para carregar as imagens.
  - Conversão para cinza, redução de tamanho, convolução, máscaras, histogramas,
    PSNR etc. são todos implementados à mão com NumPy.
  - Na Questão 2 usa-se np.fft.fft2 / ifft2 / fftshift (permitido pelo enunciado).
  - matplotlib é usado só para plotar as figuras do relatório.

Para trocar as imagens, edite o dicionário IMAGENS (arquivo e rótulo).
"""
import json
import os

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAIDA = "saida"
os.makedirs(SAIDA, exist_ok=True)

LADO_MAX = 512  # maior lado das imagens processadas

IMAGENS = {
    "A": {"arquivo": "imagem_A_ia.jpg", "rotulo": "Imagem A (gerada por IA)"},
    "B": {"arquivo": "imagem_B_humana.jpg", "rotulo": "Imagem B (pintura humana)"},
    "C": {"arquivo": "imagem_C_humana.jpg", "rotulo": "Imagem C (pintura humana)"},
}

# --------------------------------------------------------------------------
# Utilitários de imagem (tudo manual)
# --------------------------------------------------------------------------


def carregar_rgb(caminho):
    return np.asarray(Image.open(caminho).convert("RGB"), dtype=np.float64)


def para_cinza(rgb):
    """Luminância ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def reduzir(img, lado_max):
    """Redução por média de blocos (área), preservando a proporção."""
    h, w = img.shape
    esc = lado_max / max(h, w)
    if esc >= 1:
        return img
    nh, nw = int(round(h * esc)), int(round(w * esc))
    ly = np.linspace(0, h, nh + 1).astype(int)
    lx = np.linspace(0, w, nw + 1).astype(int)
    soma = np.add.reduceat(np.add.reduceat(img, ly[:-1], axis=0), lx[:-1], axis=1)
    return soma / np.outer(np.diff(ly), np.diff(lx))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(255.0**2 / m))


def histograma(img):
    return np.bincount(np.clip(np.rint(img), 0, 255).astype(np.uint8).ravel(), minlength=256)


# --------------------------------------------------------------------------
# Questão 1 - convolução 2D manual
# --------------------------------------------------------------------------

KERNELS = {
    "h1": np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], float),
    "h2": np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], float) / 256.0,
    "h3": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "h4": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "h5": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float),
    "h6": np.ones((3, 3)) / 9.0,
    "h7": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], float),
    "h8": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], float),
    "h9": np.eye(9) / 9.0,
    "h10": np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1],
                     [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]], float) / 8.0,
    "h11": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], float),
}

DESCRICAO = {
    "h1": "Laplaciano/LoG 5×5 (bordas)",
    "h2": "Gaussiano 5×5 (suavização)",
    "h3": "Sobel vertical (bordas |)",
    "h4": "Sobel horizontal (bordas —)",
    "h5": "Laplaciano 8-viz. (bordas)",
    "h6": "Média 3×3 (suavização)",
    "h7": "Linhas a 45° (/)",
    "h8": "Linhas a 135° (\\)",
    "h9": "Borrão de movimento diag. 9×9",
    "h10": "Realce de nitidez 5×5",
    "h11": "Relevo (emboss) diagonal",
}

# como exibir a resposta: 'clip' (filtros de soma 1), 'abs' (bordas), 'emboss' (resposta + 128)
MODO_EXIB = {"h1": "abs", "h2": "clip", "h3": "abs", "h4": "abs", "h5": "abs", "h6": "clip",
             "h7": "abs", "h8": "abs", "h9": "clip", "h10": "clip", "h11": "emboss"}


def convolucao2d(img, kernel):
    """Convolução 2D verdadeira (kernel espelhado nos dois eixos).
    Borda tratada por replicação dos pixels de borda (padding manual por índices)."""
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape
    ys = np.clip(np.arange(-ph, h + ph), 0, h - 1)
    xs = np.clip(np.arange(-pw, w + pw), 0, w - 1)
    pad = img[np.ix_(ys, xs)]
    saida = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            saida += k[i, j] * pad[i:i + h, j:j + w]
    return saida


def convolucao2d_ingenua(img, kernel):
    """Versão com 4 laços (somatório da definição) usada só para validar a rápida."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape
    saida = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            s = 0.0
            for m in range(kh):
                for n in range(kw):
                    yy = min(max(y - (m - ph), 0), h - 1)
                    xx = min(max(x - (n - pw), 0), w - 1)
                    s += kernel[m, n] * img[yy, xx]
            saida[y, x] = s
    return saida


def exibir_resposta(resp, modo):
    if modo == "clip":
        return np.clip(resp, 0, 255)
    if modo == "emboss":
        return np.clip(resp + 128, 0, 255)
    a = np.abs(resp)
    lim = np.percentile(a, 99.5)
    return np.clip(a / lim * 255, 0, 255) if lim > 0 else a


def validar_convolucao(img):
    rec = img[100:116, 100:116]
    for nome, k in KERNELS.items():
        a = convolucao2d(rec, k)
        b = convolucao2d_ingenua(rec, k)
        assert np.allclose(a, b), f"convolução diverge em {nome}"
    print("Validação da convolução (rápida vs. definição com laços): OK")


# --------------------------------------------------------------------------
# Questão 2 - domínio da frequência
# --------------------------------------------------------------------------


def fft_centrada(img):
    return np.fft.fftshift(np.fft.fft2(img))


def ifft_de_centrada(Fs):
    return np.real(np.fft.ifft2(np.fft.ifftshift(Fs)))


def distancias(shape):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)


def mascara(shape, tipo, r1, r2=None):
    D = distancias(shape)
    if tipo == "pb":
        return (D <= r1).astype(float)
    if tipo == "pa":
        return (D > r1).astype(float)
    if tipo == "pf":
        return ((D >= r1) & (D <= r2)).astype(float)
    if tipo == "rf":
        return (~((D >= r1) & (D <= r2))).astype(float)
    raise ValueError(tipo)


def log_espectro(Fs):
    return np.log1p(np.abs(Fs))


def exibir_filtrada(rec, tipo):
    if tipo in ("pb", "rf"):
        return np.clip(rec, 0, 255)
    a = np.abs(rec)
    lim = np.percentile(a, 99.5)
    return np.clip(a / lim * 255, 0, 255) if lim > 0 else a


def comprimir(Fs, frac):
    """Zera coeficientes com |F| < T, onde T é escolhido para manter a fração `frac`."""
    mag = np.abs(Fs)
    T = float(np.percentile(mag, 100 * (1 - frac)))
    M = mag >= T
    rec = np.clip(ifft_de_centrada(Fs * M), 0, 255)
    return rec, T, int(M.sum())


def perfil_radial(Fs):
    h, w = Fs.shape
    P = np.abs(Fs) ** 2
    D = distancias(Fs.shape).astype(int)
    rmax = min(h, w) // 2
    soma = np.bincount(D.ravel(), weights=P.ravel())
    cnt = np.bincount(D.ravel())
    prof = soma[:rmax] / cnt[:rmax]
    f = np.arange(rmax) / min(h, w)
    return f, prof


def fracao_energia_alta(Fs, f_corte):
    P = np.abs(Fs) ** 2
    D = distancias(Fs.shape)
    P_sem_dc = P.copy()
    P_sem_dc[Fs.shape[0] // 2, Fs.shape[1] // 2] = 0
    return float(P_sem_dc[D > f_corte * min(Fs.shape)].sum() / P_sem_dc.sum())


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------


def mostrar(ax, im, titulo, fs=8, vmin=0, vmax=255):
    ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_title(titulo, fontsize=fs)
    ax.axis("off")


def salvar(fig, nome):
    caminho = os.path.join(SAIDA, nome)
    fig.savefig(caminho, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def fig_imagens(rgbs, cinzas):
    fig, axs = plt.subplots(2, 3, figsize=(13, 9))
    for j, ch in enumerate(IMAGENS):
        axs[0, j].imshow(rgbs[ch].astype(np.uint8))
        axs[0, j].set_title(f"{IMAGENS[ch]['rotulo']}\noriginal {rgbs[ch].shape[1]}×{rgbs[ch].shape[0]}", fontsize=9)
        axs[0, j].axis("off")
        g = cinzas[ch]
        mostrar(axs[1, j], g, f"cinza redimensionada {g.shape[1]}×{g.shape[0]}", fs=9)
    return salvar(fig, "00_imagens.png")


def fig_q1(ch, gray, resp):
    fig, axs = plt.subplots(3, 4, figsize=(13, 10.5))
    mostrar(axs.flat[0], gray, "original (cinza)", fs=9)
    for ax, nome in zip(axs.flat[1:], KERNELS):
        mostrar(ax, exibir_resposta(resp[nome], MODO_EXIB[nome]), f"{nome} · {DESCRICAO[nome]}", fs=8.5)
    fig.suptitle(f"Questão 1 – {IMAGENS[ch]['rotulo']}", fontsize=12)
    return salvar(fig, f"q1_{ch}.png")


def fig_q2_base(ch, gray, Fs):
    rec = ifft_de_centrada(Fs)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
    mostrar(axs[0], gray, "imagem original (cinza)", fs=9)
    mostrar(axs[1], log_espectro(Fs), "espectro de Fourier (log da magnitude)", fs=9, vmin=None, vmax=None)
    mostrar(axs[2], np.clip(rec, 0, 255), f"após FFT inversa (erro máx. {np.abs(rec - gray).max():.1e})", fs=9)
    fig.suptitle(f"{IMAGENS[ch]['rotulo']}", fontsize=11)
    return salvar(fig, f"q2_base_{ch}.png")


REF_R = 30
REF_BANDA = (30, 80)


def fig_q2_nucleos(ch, gray, Fs):
    specs = [("pb", "passa-baixa", (REF_R, None)), ("pa", "passa-alta", (REF_R, None)),
             ("pf", "passa-faixa", REF_BANDA), ("rf", "rejeita-faixa", REF_BANDA)]
    fig, axs = plt.subplots(3, 4, figsize=(13, 10.5))
    L = log_espectro(Fs)
    for j, (tipo, nome, (r1, r2)) in enumerate(specs):
        M = mascara(gray.shape, tipo, r1, r2)
        rec = ifft_de_centrada(Fs * M)
        par = f"r={r1}" if r2 is None else f"r={r1}–{r2}"
        mostrar(axs[0, j], M, f"máscara {nome} ({par})", fs=9, vmin=0, vmax=1)
        mostrar(axs[1, j], L * M, "espectro × máscara", fs=9, vmin=0, vmax=L.max())
        mostrar(axs[2, j], exibir_filtrada(rec, tipo), f"imagem após filtro {nome}", fs=9)
    fig.suptitle(f"Questão 2 – núcleos e resultados · {IMAGENS[ch]['rotulo']}", fontsize=12)
    return salvar(fig, f"q2_nucleos_{ch}.png")


RAIOS = [8, 25, 60]
BANDAS = [(8, 25), (25, 60), (60, 120)]


def fig_q2_exp(ch, gray, Fs, metricas):
    fig, axs = plt.subplots(4, 3, figsize=(11, 14.5))
    E_total = float((np.abs(Fs) ** 2).sum())
    linhas = [("pb", "passa-baixa", RAIOS), ("pa", "passa-alta", RAIOS),
              ("pf", "passa-faixa", BANDAS), ("rf", "rejeita-faixa", BANDAS)]
    for i, (tipo, nome, params) in enumerate(linhas):
        for j, p in enumerate(params):
            r1, r2 = (p, None) if np.isscalar(p) else p
            M = mascara(gray.shape, tipo, r1, r2)
            rec = ifft_de_centrada(Fs * M)
            energia = float((np.abs(Fs * M) ** 2).sum() / E_total * 100)
            par = f"r={r1}" if r2 is None else f"r={r1}–{r2}"
            extra = f"energia retida {energia:.2f}%"
            reg = {"filtro": nome, "param": par, "energia_retida_pct": energia}
            if tipo in ("pb", "rf"):
                v = psnr(gray, np.clip(rec, 0, 255))
                extra += f" · PSNR {v:.1f} dB"
                reg["psnr_db"] = v
            metricas.setdefault(ch, []).append(reg)
            mostrar(axs[i, j], exibir_filtrada(rec, tipo), f"{nome} ({par})\n{extra}", fs=8.5)
    fig.suptitle(f"Questão 2 – variação dos parâmetros · {IMAGENS[ch]['rotulo']}", fontsize=12, y=0.995)
    fig.tight_layout()
    return salvar(fig, f"q2_exp_{ch}.png")


FRACOES = [0.25, 0.10, 0.02, 0.005]


def fig_q2_comp(ch, gray, Fs, tabela):
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.8))
    mostrar(axs[0, 0], gray, "original", fs=9)
    h0 = histograma(gray)
    axs[1, 0].bar(np.arange(256), h0, width=1.0, color="blue")
    axs[1, 0].set_title("histograma original", fontsize=9)
    for j, fr in enumerate(FRACOES, start=1):
        rec, T, n = comprimir(Fs, fr)
        v = psnr(gray, rec)
        tabela.append({"imagem": ch, "fracao_mantida_pct": fr * 100, "limiar_T": T,
                       "coef_mantidos": n, "coef_total": int(gray.size),
                       "razao_coef": gray.size / n, "psnr_db": v})
        mostrar(axs[0, j], rec, f"mantém {fr*100:g}% dos coef. (T={T:.0f})\nPSNR {v:.1f} dB", fs=9)
        axs[1, j].bar(np.arange(256), histograma(rec), width=1.0, color="blue")
        axs[1, j].set_title("histograma após compressão", fontsize=9)
    for ax in axs[1]:
        ax.set_xlim(0, 255)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Questão 2 – compressão por limiar · {IMAGENS[ch]['rotulo']}", fontsize=12)
    fig.tight_layout()
    return salvar(fig, f"q2_comp_{ch}.png")


def fig_resumo(espectros, grays):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    cores = {"A": "#d62728", "B": "#1f77b4", "C": "#2ca02c"}
    perfis = {}
    for ch, Fs in espectros.items():
        f, p = perfil_radial(Fs)
        perfis[ch] = (f, p)
        axs[0].loglog(f[1:], p[1:], color=cores[ch], label=IMAGENS[ch]["rotulo"])
    axs[0].set_xlabel("frequência radial (ciclos/pixel)")
    axs[0].set_ylabel("potência média")
    axs[0].set_title("Perfil radial do espectro de potência", fontsize=10)
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3, which="both")
    fr = np.geomspace(0.002, 0.6, 25)  # 100% daria erro ~0 (PSNR infinito) e distorce o eixo
    for ch, Fs in espectros.items():
        ps = [psnr(grays[ch], comprimir(Fs, x)[0]) for x in fr]
        axs[1].semilogx(fr * 100, ps, color=cores[ch], marker="o", ms=3, label=IMAGENS[ch]["rotulo"])
    axs[1].set_xlabel("% de coeficientes mantidos")
    axs[1].set_ylabel("PSNR (dB)")
    axs[1].set_title("Qualidade da reconstrução × compressão", fontsize=10)
    axs[1].grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return salvar(fig, "q2_resumo.png"), perfis, fr


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------


def main():
    rgbs, grays = {}, {}
    for ch, info in IMAGENS.items():
        rgb = carregar_rgb(info["arquivo"])
        rgbs[ch] = rgb
        grays[ch] = reduzir(para_cinza(rgb), LADO_MAX)
        print(ch, "original", rgb.shape[:2][::-1], "-> processada", grays[ch].shape[::-1])
    fig_imagens(rgbs, grays)
    validar_convolucao(grays["A"])

    resultados = {"q1": {}, "q2_exp": {}, "q2_comp": [], "espectral": {}, "tamanhos": {}}
    espectros = {}

    for ch, g in grays.items():
        resultados["tamanhos"][ch] = {"original": list(rgbs[ch].shape[:2][::-1]),
                                      "processada": list(g.shape[::-1])}
        # ---------------- Questão 1
        resp = {n: convolucao2d(g, k) for n, k in KERNELS.items()}
        fig_q1(ch, g, resp)
        tab = {}
        for n, k in KERNELS.items():
            r = resp[n]
            tab[n] = {"soma_kernel": float(k.sum()), "media": float(r.mean()), "desvio": float(r.std())}
            if abs(k.sum() - 1) < 1e-9:
                tab[n]["psnr_vs_original"] = psnr(g, np.clip(r, 0, 255))
        resultados["q1"][ch] = tab

        # ---------------- Questão 2
        Fs = fft_centrada(g)
        espectros[ch] = Fs
        fig_q2_base(ch, g, Fs)
        fig_q2_nucleos(ch, g, Fs)
        fig_q2_exp(ch, g, Fs, resultados["q2_exp"])
        fig_q2_comp(ch, g, Fs, resultados["q2_comp"])

        f, p = perfil_radial(Fs)
        sel = (f >= 0.02) & (f <= 0.25)
        coef = np.polyfit(np.log10(f[sel]), np.log10(p[sel]), 1)
        resultados["espectral"][ch] = {
            "expoente_beta": float(-coef[0]),
            "energia_acima_0.10": fracao_energia_alta(Fs, 0.10),
            "energia_acima_0.25": fracao_energia_alta(Fs, 0.25),
        }

    _, perfis, fr = fig_resumo(espectros, grays)
    resultados["curva_psnr_fracoes"] = list(map(float, fr))

    with open(os.path.join(SAIDA, "metricas.json"), "w") as fh:
        json.dump(resultados, fh, indent=2)
    print("Figuras e métricas salvas em", SAIDA)


if __name__ == "__main__":
    main()