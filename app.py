import io
import base64
import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')

# Central viruses list used across pages
VIRUSES = [
    {"id": 1, "name": "COVID-19", "lat": 20.0, "lng": 78.0, "desc": "Коронавирус пандемиясы (орыс)"},
    {"id": 2, "name": "Influenza", "lat": 40.7, "lng": -74.0, "desc": "Жедел тыныс алу вирустары"},
    {"id": 3, "name": "Ebola", "lat": 6.5, "lng": 20.0, "desc": "Орталық Африка аймағы"},
    {"id": 4, "name": "SARS", "lat": 22.3, "lng": 114.2, "desc": "Бауырлас SARS вирусы"},
    {"id": 5, "name": "Spanish Flu", "lat": 40.4168, "lng": -3.7038, "desc": "1918 жылғы Испания тұмауы (Мадрид көрсету)"},
    {"id": 6, "name": "HIV/AIDS", "lat": -4.4419, "lng": 15.2663, "desc": "Оңтүстік Африка/Конго аймағында кең тараған"},
    {"id": 7, "name": "Zika", "lat": -8.0476, "lng": -34.8770, "desc": "Бразилиядағы Зика таралуы"},
    {"id": 8, "name": "Cholera (Haiti)", "lat": 18.5944, "lng": -72.3074, "desc": "Гаитидегі холера өршіген орталық"}
]

# Sample local outbreaks in Kazakhstan (origin with targets)
LOCAL_OUTBREAKS = [
    {
        "region": "Almaty",
        "disease": "COVID-19",
        "lat": 43.2383,
        "lng": 76.9450,
        "cases": 420,
        "spread_to": [
            {"region": "Shymkent", "disease": "COVID-19", "lat": 42.3417, "lng": 69.5901, "cases": 120},
            {"region": "Kyzylorda", "disease": "COVID-19", "lat": 44.8489, "lng": 65.4821, "cases": 40},
            {"region": "Zhambyl (Taraz)", "disease": "COVID-19", "lat": 42.8986, "lng": 71.3667, "cases": 55}
        ]
    },
    {
        "region": "Nur-Sultan",
        "disease": "Influenza",
        "lat": 51.1605,
        "lng": 71.4704,
        "cases": 310,
        "spread_to": [
            {"region": "Pavlodar", "disease": "Influenza", "lat": 52.2978, "lng": 76.9450, "cases": 45},
            {"region": "Karaganda", "disease": "Influenza", "lat": 49.8067, "lng": 73.0851, "cases": 70}
        ]
    },
    {
        "region": "Shymkent",
        "disease": "Measles",
        "lat": 42.3417,
        "lng": 69.5901,
        "cases": 210,
        "spread_to": [
            {"region": "Turkistan", "disease": "Measles", "lat": 43.3136, "lng": 68.2599, "cases": 80},
            {"region": "Kyzylorda", "disease": "Measles", "lat": 44.8489, "lng": 65.4821, "cases": 20}
        ]
    },
    {
        "region": "Aktobe",
        "disease": "Cholera",
        "lat": 50.2839,
        "lng": 57.1669,
        "cases": 95,
        "spread_to": [
            {"region": "Oral", "disease": "Cholera", "lat": 51.2181, "lng": 51.3597, "cases": 25}
        ]
    },
    {
        "region": "Aktau",
        "disease": "Gastroenteritis",
        "lat": 43.6542,
        "lng": 51.2000,
        "cases": 60,
        "spread_to": [
            {"region": "Zhanaozen", "disease": "Gastroenteritis", "lat": 43.3333, "lng": 52.8250, "cases": 18}
        ]
    },
    {
        "region": "Atyrau",
        "disease": "Influenza",
        "lat": 47.1004,
        "lng": 51.8796,
        "cases": 130,
        "spread_to": [
            {"region": "Aktobe", "disease": "Influenza", "lat": 50.2839, "lng": 57.1669, "cases": 35}
        ]
    },
    {
        "region": "Karaganda",
        "disease": "COVID-19",
        "lat": 49.8067,
        "lng": 73.0851,
        "cases": 180,
        "spread_to": [
            {"region": "Pavlodar", "disease": "COVID-19", "lat": 52.2978, "lng": 76.9450, "cases": 40},
            {"region": "Kostanay", "disease": "COVID-19", "lat": 53.2194, "lng": 63.6246, "cases": 22}
        ]
    },
    {
        "region": "Pavlodar",
        "disease": "Influenza",
        "lat": 52.2978,
        "lng": 76.9450,
        "cases": 75,
        "spread_to": [
            {"region": "Semey (Öskemen)", "disease": "Influenza", "lat": 49.9688, "lng": 82.6141, "cases": 30}
        ]
    },
    {
        "region": "Kostanay",
        "disease": "Diphtheria",
        "lat": 53.2194,
        "lng": 63.6246,
        "cases": 40,
        "spread_to": [
            {"region": "Petropavl", "disease": "Diphtheria", "lat": 54.8741, "lng": 69.1606, "cases": 12}
        ]
    },
    {
        "region": "Kyzylorda",
        "disease": "COVID-19",
        "lat": 44.8489,
        "lng": 65.4821,
        "cases": 68,
        "spread_to": [
            {"region": "Shymkent", "disease": "COVID-19", "lat": 42.3417, "lng": 69.5901, "cases": 22}
        ]
    },
    {
        "region": "Taraz",
        "disease": "Measles",
        "lat": 42.8986,
        "lng": 71.3667,
        "cases": 55,
        "spread_to": [
            {"region": "Almaty", "disease": "Measles", "lat": 43.2383, "lng": 76.9450, "cases": 10}
        ]
    },
    {
        "region": "Petropavl",
        "disease": "COVID-19",
        "lat": 54.8741,
        "lng": 69.1606,
        "cases": 48,
        "spread_to": [
            {"region": "Kostanay", "disease": "COVID-19", "lat": 53.2194, "lng": 63.6246, "cases": 8}
        ]
    },
    {
        "region": "Turkistan",
        "disease": "Cholera",
        "lat": 43.3136,
        "lng": 68.2599,
        "cases": 90,
        "spread_to": [
            {"region": "Shymkent", "disease": "Cholera", "lat": 42.3417, "lng": 69.5901, "cases": 30}
        ]
    }
]


# -------------------------------------------------
# 1. Single Simulation
# -------------------------------------------------
def run_single_simulation(beta, gamma, days, N, I0, seed, sim_id):
    np.random.seed(seed)

    S = np.zeros(days, dtype=int)
    I = np.zeros(days, dtype=int)
    R = np.zeros(days, dtype=int)

    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    for t in range(1, days):
        if S[t - 1] > 0 and I[t - 1] > 0:
            inf_prob = 1 - np.exp(-beta * I[t - 1] / N)
            inf_prob = np.clip(inf_prob, 0, 1)
            new_inf = np.random.binomial(S[t - 1], inf_prob)
        else:
            new_inf = 0

        if I[t - 1] > 0:
            rec_prob = 1 - np.exp(-gamma)
            rec_prob = np.clip(rec_prob, 0, 1)
            new_rec = np.random.binomial(I[t - 1], rec_prob)
        else:
            new_rec = 0

        S[t] = S[t - 1] - new_inf
        I[t] = I[t - 1] + new_inf - new_rec
        R[t] = R[t - 1] + new_rec

    df = pd.DataFrame({
        "day": np.arange(days),
        "S": S, "I": I, "R": R,
        "simulation": sim_id
    })

    return df


# ProcessPool-friendly wrapper
def unpack_and_run(args):
    return run_single_simulation(*args)


# -------------------------------------------------
# 2. Parallel Simulations
# -------------------------------------------------
def run_parallel_simulations(beta, gamma, days, N, I0, n_sims):
    args_list = []
    for sim_id in range(n_sims):
        seed = 123 + sim_id
        args_list.append((beta, gamma, days, N, I0, seed, sim_id))

    results = []
    # Use ThreadPoolExecutor to avoid Windows pickling issues with ProcessPool
    with ThreadPoolExecutor() as executor:
        for df in executor.map(unpack_and_run, args_list):
            results.append(df)

    return pd.concat(results, ignore_index=True)


# -------------------------------------------------
# 3. Analyze Results
# -------------------------------------------------
def analyze_results(big_df):
    summary_rows = []

    for sim_id, df in big_df.groupby("simulation"):
        idx = df["I"].idxmax()
        summary_rows.append({
            "simulation": sim_id,
            "peak_day": int(df.loc[idx, "day"]),
            "peak_infected": int(df.loc[idx, "I"])
        })

    summary_df = pd.DataFrame(summary_rows)

    stats = {
        "avg_peak_infected": summary_df["peak_infected"].mean(),
        "std_peak_infected": summary_df["peak_infected"].std(),
        "avg_peak_day": summary_df["peak_day"].mean(),
        "final_mean_recovered": big_df.groupby("simulation")["R"].last().mean()
    }

    return summary_df, stats


# -------------------------------------------------
# 4. Create Chart
# -------------------------------------------------
def create_plot(big_df):
    plt.figure(figsize=(8,5))

    for sim_id, df in big_df.groupby("simulation"):
        plt.plot(df["day"], df["I"], alpha=0.25)

    mean_df = big_df.groupby("day")[["I"]].mean()
    plt.plot(mean_df.index, mean_df["I"], linewidth=3, color="blue")

    plt.title("Параллель эпидемия симуляциясы")
    plt.xlabel("Күн")
    plt.ylabel("Жұққандар саны")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # return base64 by default for backward compatibility
    return base64.b64encode(buf.getvalue()).decode()


def create_plot_file(big_df):
    """Create PNG from dataframe and save to static/plots, returning the static URL."""
    # create figure similarly to create_plot
    plt.figure(figsize=(8,5))
    for sim_id, df in big_df.groupby("simulation"):
        plt.plot(df["day"], df["I"], alpha=0.25)

    mean_df = big_df.groupby("day")[['I']].mean()
    plt.plot(mean_df.index, mean_df['I'], linewidth=3, color="blue")

    plt.title("Параллель эпидемия симуляциясы")
    plt.xlabel("Күн")
    plt.ylabel("Жұққандар саны")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    plots_dir = os.path.join(app.root_path, 'static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(plots_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(buf.getvalue())

    return url_for('static', filename=f'plots/{filename}')


# -------------------------------------------------
# 5. ROUTES
# -------------------------------------------------

# ----------- Home Page -----------
@app.route("/")
def home():
    return render_template("index.html", active="home")


# ----------- Simulation Page -----------
@app.route("/simulation", methods=["GET", "POST"])
def simulation():
    params = {
        "population": 1000,
        "initial_infected": 10,
        "beta": 0.3,
        "gamma": 0.1,
        "days": 100,
        "n_sims": 8
    }

    plot_url = None
    summary_table = None
    stats = None
    # use central viruses list
    viruses = VIRUSES

    if request.method == "POST":
        params["population"] = int(request.form["population"])
        params["initial_infected"] = int(request.form["initial_infected"])
        params["beta"] = float(request.form["beta"])
        params["gamma"] = float(request.form["gamma"])
        params["days"] = int(request.form["days"])
        params["n_sims"] = int(request.form["n_sims"])

        big_df = run_parallel_simulations(
            params["beta"], params["gamma"],
            params["days"], params["population"],
            params["initial_infected"], params["n_sims"]
        )

        summary_df, stats = analyze_results(big_df)
        summary_table = summary_df.to_html(classes="table table-bordered text-center", index=False)

        # persist summary and stats into session so the table remains across pages
        try:
            session['summary_table'] = summary_table
            session['stats'] = {k: float(v) for k, v in stats.items()}
        except Exception:
            session.pop('summary_table', None)
            session.pop('stats', None)

        # persist summary and stats into session so the table remains across pages
        try:
            session['summary_table'] = summary_table
            session['stats'] = {k: float(v) for k, v in stats.items()}
        except Exception:
            session.pop('summary_table', None)
            session.pop('stats', None)

        # try to save a persistent plot file for cross-page persistence
        try:
            plot_file = create_plot_file(big_df)
            session['plot_file'] = plot_file
            plot_url = None
        except Exception:
            plot_url = create_plot(big_df)
            session.pop('plot_file', None)

    # prefer persistent file and data from session when available
    plot_file = session.get('plot_file')
    if not summary_table:
        summary_table = session.get('summary_table')
    if not stats:
        stats = session.get('stats')
    if not summary_table:
        summary_table = session.get('summary_table')
    if not stats:
        stats = session.get('stats')
    return render_template("simulation.html",
                           params=params,
                           plot_url=plot_url,
                           plot_file=plot_file,
                           summary_table=summary_table,
                           stats=stats,
                           viruses=viruses,
                           active="simulation")


# ----------- Stats Page -----------
@app.route("/stats", methods=["GET", "POST"])
def stats_page():
    params = {
        "population": 1000,
        "initial_infected": 10,
        "beta": 0.3,
        "gamma": 0.1,
        "days": 100,
        "n_sims": 8
    }

    plot_url = None
    summary_table = None
    stats = None

    if request.method == "POST":
        params["population"] = int(request.form["population"])
        params["initial_infected"] = int(request.form["initial_infected"])
        params["beta"] = float(request.form["beta"])
        params["gamma"] = float(request.form["gamma"])
        params["days"] = int(request.form["days"])
        params["n_sims"] = int(request.form["n_sims"])

        big_df = run_parallel_simulations(
            params["beta"], params["gamma"],
            params["days"], params["population"],
            params["initial_infected"], params["n_sims"]
        )

        summary_df, stats = analyze_results(big_df)
        summary_table = summary_df.to_html(classes="table table-bordered text-center", index=False)

        # try to save persistent file for cross-page persistence
        try:
            plot_file = create_plot_file(big_df)
            session['plot_file'] = plot_file
            plot_url = None
        except Exception:
            plot_url = create_plot(big_df)
            session.pop('plot_file', None)
    plot_file = session.get('plot_file')
    return render_template("stats.html",
                           params=params,
                           plot_url=plot_url,
                           plot_file=plot_file,
                           summary_table=summary_table,
                           stats=stats,
                           active="stats")


# ----------- Local Outbreaks Page -----------
@app.route('/local')
def local_outbreaks():
    return render_template('local_outbreaks.html', viruses=VIRUSES, local_outbreaks=LOCAL_OUTBREAKS, active='local')


if __name__ == "__main__":
    app.run(debug=True)
