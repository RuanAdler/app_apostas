
# -*- coding: utf-8 -*-
import math
import sqlite3
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

DB_PATH = "aposta.db"

# -----------------------------
# Utils & DB
# -----------------------------
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS parameters ("
        "id INTEGER PRIMARY KEY CHECK (id=1),"
        "bank REAL DEFAULT 1000.0,"
        "stake_pct REAL DEFAULT 1.0,"
        "kelly_frac REAL DEFAULT 0.25,"
        "margin_fair REAL DEFAULT 3.0,"
        "odds_min_corners REAL DEFAULT 1.65,"
        "odds_min_goal2t REAL DEFAULT 1.70"
        ")"
    )
    cur.execute("INSERT OR IGNORE INTO parameters (id) VALUES (1)")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS pregame ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "gdate TEXT, league TEXT, home TEXT, away TEXT,"
        "xgf_home REAL, xga_home REAL, xgf_away REAL, xga_away REAL,"
        "xg_league_home REAL, xg_league_away REAL,"
        "odds_mkt_o25 REAL, odds_mkt_btts REAL,"
        "notes TEXT"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS live_corners ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "gdate TEXT, league TEXT, home TEXT, away TEXT,"
        "minute INTEGER, attacks INTEGER, sot INTEGER, shots INTEGER,"
        "corners INTEGER, red_card TEXT, odds_mkt REAL, notes TEXT"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS live_goal2t ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "gdate TEXT, league TEXT, home TEXT, away TEXT,"
        "minute INTEGER, score TEXT, fav_losing TEXT, draw TEXT,"
        "sot INTEGER, shots INTEGER, attacks10 INTEGER, red_card TEXT,"
        "odds_mkt REAL, notes TEXT"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS lay_fav ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "gdate TEXT, league TEXT, home TEXT, away TEXT,"
        "minute INTEGER, odd_fav REAL, pos_fav REAL, sot_fav INTEGER, sot_adv INTEGER,"
        "odd_lay REAL, volume REAL, notes TEXT"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "ts TEXT,"
        "gdate TEXT, league TEXT, home TEXT, away TEXT,"
        "market TEXT, minute INTEGER,"
        "odds_entry REAL, odds_close REAL,"
        "prob REAL, ev REAL, stake REAL,"
        "result TEXT, profit REAL, clv REAL,"
        "source TEXT, flags TEXT, link TEXT"
        ")"
    )
    conn.commit()
    return conn

def get_parameters(conn):
    row = conn.execute("SELECT bank, stake_pct, kelly_frac, margin_fair, odds_min_corners, odds_min_goal2t FROM parameters WHERE id=1").fetchone()
    return dict(bank=row[0], stake_pct=row[1], kelly_frac=row[2], margin_fair=row[3], odds_min_corners=row[4], odds_min_goal2t=row[5])

def update_parameters(conn, **kwargs):
    cols = ["bank","stake_pct","kelly_frac","margin_fair","odds_min_corners","odds_min_goal2t"]
    sets, vals = [], []
    for c in cols:
        if c in kwargs and kwargs[c] is not None:
            sets.append(f"{c}=?")
            vals.append(kwargs[c])
    if sets:
        vals.append(1)
        conn.execute(f"UPDATE parameters SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()

def bank_current(conn)->float:
    bank0 = conn.execute("SELECT bank FROM parameters WHERE id=1").fetchone()[0]
    profit = conn.execute("SELECT COALESCE(SUM(profit),0) FROM history").fetchone()[0]
    return bank0 + profit

# -----------------------------
# Math
# -----------------------------
def poisson_over25_prob(lmb_total: float) -> float:
    cdf = 0.0
    for i in range(0, 3):
        cdf += math.exp(-lmb_total) * (lmb_total**i) / math.factorial(i)
    return 1 - cdf

def btts_prob(lmb_home: float, lmb_away: float) -> float:
    return 1 - math.exp(-lmb_home) - math.exp(-lmb_away) + math.exp(-(lmb_home + lmb_away))

def fair_odds(p: float) -> Optional[float]:
    if p<=0: return None
    return 1.0/p

def ev_decimal(prob: float, odds: Optional[float]) -> Optional[float]:
    if not odds or odds<=1: return None
    return prob*odds - 1.0

def kelly_fraction(odds: float, p: float) -> float:
    b = odds - 1.0
    q = 1.0 - p
    if b<=0: return 0.0
    f = (b*p - q) / b
    return max(0.0, f)

def clv_percent(odds_entry: float, odds_close: Optional[float]) -> Optional[float]:
    if not odds_close or odds_close<=0 or not odds_entry or odds_entry<=0:
        return None
    return ((1/odds_close) - (1/odds_entry)) / (1/odds_close)

# -----------------------------
# UI helpers
# -----------------------------
st.set_page_config(page_title="Apostas PRO - App", layout="wide")

def header_badge(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

def stake_min(conn, odds: float, prob: float) -> float:
    p = get_parameters(conn)
    bank = bank_current(conn)
    fixed = p["stake_pct"]/100.0 * bank
    kfrac = p["kelly_frac"] * kelly_fraction(odds, prob) * bank
    return round(min(fixed, kfrac if kfrac>0 else fixed), 2)

# -----------------------------
# App
# -----------------------------
conn = init_db()
params = get_parameters(conn)

with st.sidebar:
    st.title("⚙️ Parâmetros")
    bank = st.number_input("Banca Inicial (R$)", min_value=0.0, value=float(params["bank"]), step=100.0, format="%.2f")
    stake_pct = st.number_input("Stake %", min_value=0.0, value=float(params["stake_pct"]), step=0.25, format="%.2f")
    kelly_frac = st.number_input("Kelly fracionado (0-1)", min_value=0.0, max_value=1.0, value=float(params["kelly_frac"]), step=0.05, format="%.2f")
    margin_fair = st.number_input("Margem Fair (%)", min_value=0.0, value=float(params["margin_fair"]), step=0.5, format="%.2f")
    odds_min_corners = st.number_input("Odds mín. Cantos", min_value=1.01, value=float(params["odds_min_corners"]), step=0.01, format="%.2f")
    odds_min_goal2t = st.number_input("Odds mín. Gol 2ºT", min_value=1.01, value=float(params["odds_min_goal2t"]), step=0.01, format="%.2f")
    if st.button("Salvar parâmetros"):
        update_parameters(conn, bank=bank, stake_pct=stake_pct, kelly_frac=kelly_frac,
                          margin_fair=margin_fair, odds_min_corners=odds_min_corners, odds_min_goal2t=odds_min_goal2t)
        st.success("Parâmetros salvos.")
    st.divider()
    st.metric("Banca Atual (estimada)", f"R$ {bank_current(conn):,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

tabs = st.tabs(["Pré-Jogo", "Live – Escanteios", "Live – Gol 2º Tempo", "Lay Favorito", "Histórico", "Dashboard"])

# -----------------------------
# Pré-Jogo
# -----------------------------
with tabs[0]:
    header_badge("Pré-Jogo", "Poisson de xG + EV/odds justas")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gdate = st.date_input("Data", value=date.today())
        league = st.text_input("Liga", value="BRA1")
    with col2:
        home = st.text_input("Mandante", value="Flamengo")
        away = st.text_input("Visitante", value="Palmeiras")
    with col3:
        xgf_home = st.number_input("xGF Home (casa)", min_value=0.0, value=1.85, step=0.05)
        xga_home = st.number_input("xGA Home (casa)", min_value=0.0, value=1.10, step=0.05)
        xgf_away = st.number_input("xGF Away (fora)", min_value=0.0, value=1.30, step=0.05)
        xga_away = st.number_input("xGA Away (fora)", min_value=0.0, value=1.20, step=0.05)
    with col4:
        xg_l_home = st.number_input("xG Liga (casa)", min_value=0.0, value=1.55, step=0.05)
        xg_l_away = st.number_input("xG Liga (fora)", min_value=0.0, value=1.35, step=0.05)
        odds_mkt_o25 = st.number_input("Odds Mercado Over 2.5", min_value=1.0, value=1.95, step=0.01)
        odds_mkt_btts = st.number_input("Odds Mercado BTTS", min_value=1.0, value=1.95, step=0.01)
    notes = st.text_input("Observações", value="")

    # Lambdas
    l_home = xg_l_home * (xgf_home/xg_l_home) * (xga_away/xg_l_away) if xg_l_home>0 and xg_l_away>0 else 0.0
    l_away = xg_l_away * (xgf_away/xg_l_away) * (xga_home/xg_l_home) if xg_l_home>0 and xg_l_away>0 else 0.0
    l_total = l_home + l_away

    p_over25 = poisson_over25_prob(l_total)
    p_btts = btts_prob(l_home, l_away)
    fair_o25 = fair_odds(p_over25)
    fair_btts = fair_odds(p_btts)
    ev_o25 = ev_decimal(p_over25, odds_mkt_o25)
    ev_btts = ev_decimal(p_btts, odds_mkt_btts)

    st.write(f"**λ_home:** {l_home:.2f} | **λ_away:** {l_away:.2f} | **λ_total:** {l_total:.2f}")
    st.write(f"**P(Over 2.5):** {p_over25:.3f} | **Odds Justas:** {fair_o25:.2f if fair_o25 else float('nan')} | **EV:** {ev_o25:.3f}")
    st.write(f"**P(BTTS):** {p_btts:.3f} | **Odds Justas:** {fair_btts:.2f if fair_btts else float('nan')} | **EV:** {ev_btts:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Salvar pré-jogo"):
            sql = ("INSERT INTO pregame (gdate,league,home,away,xgf_home,xga_home,xgf_away,xga_away,"
                   "xg_league_home,xg_league_away,odds_mkt_o25,odds_mkt_btts,notes) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (gdate.isoformat(), league, home, away, xgf_home, xga_home, xgf_away, xga_away,
                               xg_l_home, xg_l_away, odds_mkt_o25, odds_mkt_btts, notes))
            conn.commit()
            st.success("Pré-jogo salvo.")
    with c2:
        if st.button("Ver pré-jogos salvos"):
            df = pd.read_sql_query("SELECT * FROM pregame ORDER BY id DESC", conn)
            st.dataframe(df, use_container_width=True)

# -----------------------------
# Live – Escanteios
# -----------------------------
with tabs[1]:
    header_badge("Live – Escanteios", "Gatilhos: ataques≥min, SOT≥4, cantos≤5, sem vermelho + odd ≥ mínima")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gdate = st.date_input("Data (live)", value=date.today(), key="cdate")
        league = st.text_input("Liga", value="BRA1", key="cliga")
    with col2:
        home = st.text_input("Mandante", value="Flamengo", key="chome")
        away = st.text_input("Visitante", value="Palmeiras", key="caway")
    with col3:
        minute = st.number_input("Minuto", min_value=0, value=33, step=1, key="cmin")
        attacks = st.number_input("Ataques Totais", min_value=0, value=37, step=1, key="catk")
        sot = st.number_input("SOT (no alvo)", min_value=0, value=5, step=1, key="csot")
    with col4:
        shots = st.number_input("Finalizações Totais", min_value=0, value=12, step=1, key="cshots")
        corners = st.number_input("Cantos até agora", min_value=0, value=5, step=1, key="ccorners")
        red = st.selectbox("Cartão vermelho?", ["NAO","SIM"], index=0, key="cred")
        odds_mkt = st.number_input("Odd Mercado (+1,5)", min_value=1.0, value=1.68, step=0.01, key="codd")

    p = get_parameters(conn)
    gatilhos_ok = (attacks >= minute) and (sot >= 4) and (corners <= 5) and (red != "SIM")
    rec = gatilhos_ok and (odds_mkt >= p["odds_min_corners"])

    st.info(f"Odds mínimas configuradas: {p['odds_min_corners']:.2f}")
    st.write(f"**Gatilhos OK?** {'SIM' if gatilhos_ok else 'NAO'}")
    st.write(f"**Recomendação:** {'+1,5 cantos – ENTRAR' if rec else 'NÃO ENTRAR'}")

    prob_proxy = st.slider("Sua probabilidade (proxy) para +1,5 cantos", min_value=0.40, max_value=0.80, value=0.62, step=0.01)
    stake_sug = stake_min(conn, odds_mkt, prob_proxy)
    st.metric("Stake sugerida (R$)", f"{stake_sug:.2f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Salvar linha live"):
            sql = ("INSERT INTO live_corners "
                   "(gdate,league,home,away,minute,attacks,sot,shots,corners,red_card,odds_mkt,notes) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (gdate.isoformat(), league, home, away, minute, attacks, sot, shots, corners, red, odds_mkt, ""))
            conn.commit()
            st.success("Linha live salva.")
    with c2:
        if st.button("Registrar no Histórico (como entrada)"):
            ev = ev_decimal(prob_proxy, odds_mkt)
            stake = stake_sug
            ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            sql = ("INSERT INTO history "
                   "(ts,gdate,league,home,away,market,minute,odds_entry,odds_close,prob,ev,stake,result,profit,clv,source,flags,link) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (ts, gdate.isoformat(), league, home, away, "Cantos +1,5", minute, odds_mkt, None,
                               prob_proxy, ev, stake, None, None, None, "App", "AUTO", ""))
            conn.commit()
            st.success("Entrada registrada no Histórico. Depois marque WIN/LOSS e (se tiver) a odd de fechamento para medir CLV.")

    st.subheader("Últimas linhas live (salvas)")
    dfc = pd.read_sql_query("SELECT * FROM live_corners ORDER BY id DESC LIMIT 20", conn)
    st.dataframe(dfc, use_container_width=True)

# -----------------------------
# Live – Gol 2º Tempo
# -----------------------------
with tabs[2]:
    header_badge("Live – Gol 2º Tempo", "Gatilhos: min≥60, empate ou favorito perdendo, SOT≥5, Fin≥14, Atk10≥12, sem vermelho + odd ≥ mín")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        gdate = st.date_input("Data (live)", value=date.today(), key="gdate")
        league = st.text_input("Liga", value="ENG1", key="gliga")
    with col2:
        home = st.text_input("Mandante", value="Arsenal", key="ghome")
        away = st.text_input("Visitante", value="Newcastle", key="gaway")
    with col3:
        minute = st.number_input("Minuto", min_value=0, value=70, step=1, key="gmin")
        score = st.text_input("Placar", value="0-0", key="gscore")
        fav_losing = st.selectbox("Favorito perdendo?", ["NAO","SIM"], index=0, key="gfav")
    with col4:
        draw = st.selectbox("Jogo empatado?", ["SIM","NAO"], index=0, key="gdraw")
        sot = st.number_input("SOT", min_value=0, value=5, step=1, key="gsot")
        shots = st.number_input("Finalizações", min_value=0, value=15, step=1, key="gshots")
        atk10 = st.number_input("Ataques (últ.10min)", min_value=0, value=13, step=1, key="gatk10")
        red = st.selectbox("Vermelho?", ["NAO","SIM"], index=0, key="gred")
        odds_mkt = st.number_input("Odd Mercado (+0,5)", min_value=1.0, value=1.80, step=0.01, key="godd")

    p = get_parameters(conn)
    gat = (minute>=60) and ((fav_losing=="SIM") or (draw=="SIM")) and (sot>=5) and (shots>=14) and (atk10>=12) and (red!="SIM")
    rec = gat and (odds_mkt >= p["odds_min_goal2t"])
    st.info(f"Odds mínimas configuradas: {p['odds_min_goal2t']:.2f}")
    st.write(f"**Gatilhos OK?** {'SIM' if gat else 'NAO'}")
    st.write(f"**Recomendação:** {'+0,5 gol – ENTRAR' if rec else 'NÃO ENTRAR'}")

    prob_proxy = st.slider("Sua probabilidade (proxy) para +0,5 gol", min_value=0.35, max_value=0.80, value=0.57, step=0.01, key="gprob")
    stake_sug = stake_min(conn, odds_mkt, prob_proxy)
    st.metric("Stake sugerida (R$)", f"{stake_sug:.2f}")

    c1,c2 = st.columns(2)
    with c1:
        if st.button("Salvar linha live (Gol 2ºT)"):
            sql = ("INSERT INTO live_goal2t "
                   "(gdate,league,home,away,minute,score,fav_losing,draw,sot,shots,attacks10,red_card,odds_mkt,notes) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (gdate.isoformat(), league, home, away, minute, score, fav_losing, draw, sot, shots, atk10, red, odds_mkt, ""))
            conn.commit()
            st.success("Linha live (Gol 2ºT) salva.")
    with c2:
        if st.button("Registrar no Histórico (Gol 2ºT)"):
            ev = ev_decimal(prob_proxy, odds_mkt)
            stake = stake_sug
            ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            sql = ("INSERT INTO history "
                   "(ts,gdate,league,home,away,market,minute,odds_entry,odds_close,prob,ev,stake,result,profit,clv,source,flags,link) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (ts, gdate.isoformat(), league, home, away, "Gol +0,5 (2ºT)", minute, odds_mkt, None,
                               prob_proxy, ev, stake, None, None, None, "App", "AUTO", ""))
            conn.commit()
            st.success("Entrada registrada no Histórico.")

# -----------------------------
# Lay Favorito
# -----------------------------
with tabs[3]:
    header_badge("Lay Favorito", "Favorito falso: min≥25, odd<1.80, posse<47%, (SOT adv − SOT fav) ≥ 2")
    col1,col2,col3 = st.columns(3)
    with col1:
        gdate = st.date_input("Data", value=date.today(), key="ldate")
        league = st.text_input("Liga", value="ESP1", key="lliga")
        home = st.text_input("Mandante", value="Barcelona", key="lhome")
        away = st.text_input("Visitante", value="Real Sociedad", key="laway")
    with col2:
        minute = st.number_input("Minuto", min_value=0, value=33, step=1, key="lmin")
        odd_fav = st.number_input("Odd Favorito", min_value=1.0, value=1.70, step=0.01, key="lodd_fav")
        pos_fav = st.number_input("Posse Favorito (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0, key="lpos")
        sot_fav = st.number_input("SOT Favorito", min_value=0, value=1, step=1, key="lsotf")
    with col3:
        sot_adv = st.number_input("SOT Adversário", min_value=0, value=3, step=1, key="lsota")
        odd_lay = st.number_input("Odd Lay Disponível", min_value=1.0, value=1.72, step=0.01, key="lodd_lay")
        volume = st.number_input("Volume (Exchange)", min_value=0.0, value=500000.0, step=10000.0, key="lvol")

    crit = (minute>=25) and (odd_fav<1.80) and (pos_fav<47.0) and ((sot_adv - sot_fav) >= 2)
    st.write(f"**Critérios OK?** {'SIM' if crit else 'NAO'}")
    stake_expo = round(get_parameters(conn)["stake_pct"]/100.0 * bank_current(conn), 2)
    st.metric("Exposição sugerida (R$)", f"{stake_expo:.2f}")

    c1,c2 = st.columns(2)
    with c1:
        if st.button("Salvar linha Lay"):
            sql = ("INSERT INTO lay_fav "
                   "(gdate,league,home,away,minute,odd_fav,pos_fav,sot_fav,sot_adv,odd_lay,volume,notes) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (gdate.isoformat(), league, home, away, minute, odd_fav, pos_fav, sot_fav, sot_adv, odd_lay, volume, ""))
            conn.commit()
            st.success("Linha Lay salva.")
    with c2:
        if st.button("Registrar no Histórico (Lay)"):
            ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            sql = ("INSERT INTO history "
                   "(ts,gdate,league,home,away,market,minute,odds_entry,odds_close,prob,ev,stake,result,profit,clv,source,flags,link) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
            conn.execute(sql, (ts, gdate.isoformat(), league, home, away, "LAY Favorito", minute, odd_lay, None,
                               None, None, stake_expo, None, None, None, "App", "AUTO", ""))
            conn.commit()
            st.success("Entrada Lay registrada no Histórico.")

# -----------------------------
# Histórico
# -----------------------------
with tabs[4]:
    header_badge("Histórico", "Edite resultado/fechamento para calcular Lucro e CLV")
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    st.dataframe(df, use_container_width=True)
    st.caption("Dica: preencha 'odds_close' e 'result' (WIN/LOSS) para ativar CLV e Lucro.")

    with st.expander("Atualizar linha do Histórico"):
        hid = st.number_input("ID da linha", min_value=0, value=int(df["id"].iloc[0]) if len(df)>0 else 0, step=1)
        result = st.selectbox("Resultado", ["", "WIN", "LOSS"], index=0)
        odds_close = st.text_input("Odd de fechamento (opcional)", value="")
        if st.button("Salvar atualização"):
            row = conn.execute("SELECT odds_entry, stake FROM history WHERE id=?", (hid,)).fetchone()
            if not row:
                st.error("ID não encontrado.")
            else:
                odds_entry, stake = row
                result_val = result if result else None
                odds_close_val = float(odds_close.replace(",",".")) if odds_close else None
                clv = clv_percent(odds_entry, odds_close_val) if odds_close_val else None
                profit = None
                if result_val:
                    profit = stake*(odds_entry-1) if result_val=="WIN" else -stake
                conn.execute("UPDATE history SET result=?, odds_close=?, profit=?, clv=? WHERE id=?",
                             (result_val, odds_close_val, profit, clv, hid))
                conn.commit()
                st.success("Linha atualizada.")

# -----------------------------
# Dashboard
# -----------------------------
with tabs[5]:
    header_badge("Dashboard", "ROI, CLV, Hit Rate e curva de capital")
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id ASC", conn)
    if len(df)==0:
        st.info("Sem dados ainda. Registre entradas no Histórico.")
    else:
        df["profit"].fillna(0, inplace=True)
        df["stake"].fillna(0, inplace=True)
        total_stake = float(df["stake"].sum())
        roi = 100 * (df["profit"].sum() / total_stake if total_stake>0 else 0.0)
        clv_mean = 100 * (df["clv"].dropna().mean() if df["clv"].notna().sum()>0 else 0.0)
        denom = float((df["result"]=="WIN").sum() + (df["result"]=="LOSS").sum())
        hit = 100 * ((df["result"]=="WIN").sum() / denom if denom>0 else 0.0)

        c1,c2,c3 = st.columns(3)
        c1.metric("ROI (%)", f"{roi:.2f}")
        c2.metric("CLV médio (%)", f"{clv_mean:.2f}")
        c3.metric("Hit Rate (%)", f"{hit:.2f}")

        df["cum_profit"] = df["profit"].cumsum()
        bank0 = get_parameters(conn)["bank"]
        equity = bank0 + df["cum_profit"]
        plt.figure()
        plt.plot(equity.values)
        plt.title("Curva de Capital")
        plt.xlabel("Apostas")
        plt.ylabel("R$")
        st.pyplot(plt.gcf())
