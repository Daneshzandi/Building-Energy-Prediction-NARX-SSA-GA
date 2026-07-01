import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Conv1D, MaxPooling1D,Flatten, SimpleRNN, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.linalg import svd
import warnings
import os
from datetime import datetime
s = 42
np.random.seed(s)
tf.random.set_seed(s)
outd = f"Energy_base_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(outd, exist_ok=True)
df = pd.read_csv('data-set.csv')
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
t = 'Energy delta[Wh]'
f = ['GHI', 'temp', 'pressure', 'humidity', 'wind_speed','rain_1h', 'snow_1h', 'clouds_all', 'isSun', 'dayLength','weather_type', 'hour', 'month']
f = [c for c in f if c in df.columns]
data = df[f + [t]].copy().dropna()
for lag in [1, 2, 3, 6, 12, 24]:
    data[f'lag_{lag}'] = data[t].shift(lag)
data['temp_humidity'] = data['temp'] * data['humidity']
data.dropna(inplace=True)
def ssa_denoise(signal, w=30, ncomp=10):
    L = len(signal)
    K = L - w + 1
    X = np.array([signal[i:i+w] for i in range(K)]).T
    U, svals, Vt = svd(X, full_matrices=False)
    out = np.zeros(L)
    for i in range(min(ncomp, len(svals))):
        xi = svals[i] * np.outer(U[:, i], Vt[i, :])
        for j in range(L):
            idx = np.intersect1d(np.where(j - xi.shape[1] + 1 <= np.arange(xi.shape[0])),
                                 np.where(np.arange(xi.shape[0]) <= j))
            if len(idx) > 0:
                out[j] += np.mean([xi[p, j-p] for p in idx if 0<=p<xi.shape[0] and 0<=j-p<xi.shape[1]])
    return out
y_ser = data[t].values
y_den = ssa_denoise(y_ser, w=30, ncomp=10)
data['target_ssa'] = y_den
print("SSA denoisecompleted.")
cf = data[['hour', 'month']].values
scl = RobustScaler()
cfs = scl.fit_transform(cf)
km = KMeans(n_clusters=3, random_state=s, n_init=10)
data['cluster'] = km.fit_predict(cfs)
print(f"Clustering completed. Unique clusters: {np.unique(data['cluster'])}")
X_raw = data.drop(columns=[t, 'cluster']).values
y_raw = data[t].values
scX = RobustScaler()
scY = RobustScaler()
Xs = scX.fit_transform(X_raw)
Ys = scY.fit_transform(y_raw.reshape(-1,1)).flatten()
s1 = int(0.7 * len(Xs))
s2 = int(0.85 * len(Xs))
Xtr = Xs[:s1]
Ytr = Ys[:s1]
Xv = Xs[s1:s2]
Yv = Ys[s1:s2]
Xts = Xs[s2:]
Yts = Ys[s2:]
tc = data['cluster'].values[:s1]
vc = data['cluster'].values[s1:s2]
tsc = data['cluster'].values[s2:]
print(f"Train: {Xtr.shape}, Val: {Xv.shape}, Test: {Xts.shape}")

def build_cnn(shape):
    m = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=shape),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return m
def build_rnn(shape):
    m = Sequential([
        SimpleRNN(32, activation='tanh', input_shape=shape),
        Dropout(0.3),
        Dense(1)
    ])
    return m
def build_lstm(shape):
    m = Sequential([
        LSTM(32, activation='tanh', input_shape=shape),
        Dropout(0.3),
        Dense(1)
    ])
    return m
def build_cnn_lstm(shape):
    m = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=shape),
        MaxPooling1D(2),
        LSTM(32, activation='tanh'),
        Dropout(0.3),
        Dense(1)
    ])
    return m
#FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFf
def build_mlp(shape):
    m = Sequential([
        Flatten(input_shape=shape),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return m
ishape = (Xtr.shape[1], 1)
base = {
    'CNN': build_cnn(ishape),
    'RNN': build_rnn(ishape),
    'LSTM': build_lstm(ishape),
    'CNN_LSTM': build_cnn_lstm(ishape),
    'MLP': build_mlp(ishape)
}
for name, model in base.items():
    model.compile(optimizer=Adam(0.001), loss='mse')
def make_narx(dim, units=32, drop=0.2):
    m = Sequential([
        LSTM(units, input_shape=(1, dim)),
        Dropout(drop),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer=Adam(0.001), loss='mse')
    return m
def tune_narx(x_tr, y_tr, x_val, y_val):
    best_err = np.inf
    best_p = None
    for u in [32, 64]:
        for d in [0.2, 0.3]:
            mod = make_narx(x_tr.shape[1], u, d)
            es = EarlyStopping(patience=5, restore_best_weights=True)
            mod.fit(x_tr.reshape(-1,1,x_tr.shape[1]), y_tr, epochs=20, batch_size=32,
                    validation_data=(x_val.reshape(-1,1,x_val.shape[1]), y_val),
                    callbacks=[es], verbose=0)
            p = mod.predict(x_val.reshape(-1,1,x_val.shape[1]), verbose=0).flatten()
            err = np.sqrt(mean_squared_error(y_val, p))
            if err < best_err:
                best_err = err
                best_p = (u, d)
    return best_p, best_err
print("\nTrain hybrid MODEL")
hybrids = {}
best_hp = []
for cid in np.unique(tc):
    idx_tr = np.where(tc == cid)[0]
    idx_v = np.where(vc == cid)[0]
    if len(idx_tr) < 10 or len(idx_v) < 5:
        continue
    Xc_tr = Xtr[idx_tr]
    Yc_tr = Ytr[idx_tr]
    Xc_v = Xv[idx_v]
    Yc_v = Yv[idx_v]
    (u_opt, d_opt), err_opt = tune_narx(Xc_tr, Yc_tr, Xc_v, Yc_v)
    m_opt = make_narx(Xc_tr.shape[1], u_opt, d_opt)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    m_opt.fit(Xc_tr.reshape(-1,1,Xc_tr.shape[1]), Yc_tr, epochs=50, batch_size=32,
              validation_data=(Xc_v.reshape(-1,1,Xc_v.shape[1]), Yc_v),
              callbacks=[es], verbose=0)
    hybrids[cid] = m_opt
    print(f"  Cluster {cid}: units={u_opt}, dropout={d_opt}, val_RMSE={err_opt:.4f}")
    best_hp.append({'Cluster': cid, 'Units': u_opt, 'Dropout': d_opt, 'Val_RMSE': err_opt})

def evaluate(mod, X, Y, phase, model_name, scaler, is3d=False):
    if is3d:
        yp = mod.predict(X, verbose=0).flatten()
    else:
        X_in = X.reshape(-1, 1, X.shape[1])
        yp = mod.predict(X_in, verbose=0).flatten()
    yt = Y
    yt_orig = scaler.inverse_transform(yt.reshape(-1,1)).flatten()
    yp_orig = scaler.inverse_transform(yp.reshape(-1,1)).flatten()
    mse = mean_squared_error(yt_orig, yp_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt_orig, yp_orig)
    r2 = r2_score(yt_orig, yp_orig)
    mape = np.mean(np.abs((yt_orig - yp_orig) / (yt_orig + 1e-8))) * 100
    return {'phase': phase, 'model': model_name, 'RMSE': rmse, 'MSE': mse,'MAE': mae, 'R2': r2, 'MAPE': mape, 'preds': yp_orig, 'true': yt_orig}
results = []
test_preds = {}
true_test = None
print("\nEvaluate base models")
for name, model in base.items():
    print(f"  Training {name}")
    Xtr3d = Xtr.reshape(-1, Xtr.shape[1], 1)
    Xv3d = Xv.reshape(-1, Xv.shape[1], 1)
    Xts3d = Xts.reshape(-1, Xts.shape[1], 1)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(Xtr3d, Ytr, epochs=30, batch_size=32,
              validation_data=(Xv3d, Yv), callbacks=[es], verbose=0)
    for ph, Xp, Yp in [('Train', Xtr3d, Ytr), ('Validation', Xv3d, Yv), ('Test', Xts3d, Yts)]:
        r = evaluate(model, Xp, Yp, ph, name, scY, is3d=True)
        results.append(r)
        if ph == 'Test':
            test_preds[f'{name}_pred'] = r['preds']
            true_test = r['true']
def pred_hybrid(X_data, clusters, model_dict):
    preds = np.zeros(len(X_data))
    for cid, m in model_dict.items():
        idx = np.where(clusters == cid)[0]
        if len(idx) > 0:
            sub = X_data[idx]
            p = m.predict(sub.reshape(-1,1,sub.shape[1]), verbose=0).flatten()
            preds[idx] = p
    return preds
hybrid_test_preds = None
for ph, Xp, Yp, cl in [('train', Xtr, Ytr, tc),('val', Xv, Yv, vc),('test', Xts, Yts, tsc)]:
    yp = pred_hybrid(Xp, cl, hybrids)
    yt = Yp
    yt_orig = scY.inverse_transform(yt.reshape(-1,1)).flatten()
    yp_orig = scY.inverse_transform(yp.reshape(-1,1)).flatten()
    mse = mean_squared_error(yt_orig, yp_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt_orig, yp_orig)
    r2 = r2_score(yt_orig, yp_orig)
    mape = np.mean(np.abs((yt_orig - yp_orig) / (yt_orig + 1e-8))) * 100
    results.append({'phase': ph, 'model': 'Hybrid', 'RMSE': rmse, 'MSE': mse,'MAE': mae, 'R2': r2, 'MAPE': mape, 'preds': yp_orig, 'true': yt_orig})
    if ph == 'Test':
        test_preds['Hybrid_pred'] = yp_orig
        hybrid_test_preds = yp_orig
        true_test = yt_orig
metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['preds','true']} for r in results])
hp_df = pd.DataFrame(best_hp)
preds_df = pd.DataFrame({'Actual': true_test})
for name in base.keys():
    col = f'{name}_pred'
    if col in test_preds:
        preds_df[f'{name}_Pred'] = test_preds[col]
preds_df['Hybrid_Pred'] = test_preds['Hybrid_pred']
ex_path = os.path.join(outd, 'results.xlsx')
with pd.ExcelWriter(ex_path, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    hp_df.to_excel(writer, sheet_name='BestParams', index=False)
    preds_df.to_excel(writer, sheet_name='TestPredictions', index=False)
print(f"\nResults saved to: {ex_path}")
metrics_df.to_csv(os.path.join(outd, 'metrics.csv'), index=False)
base_test = metrics_df[(metrics_df['phase'] == 'Test') & (metrics_df['model'] != 'Hybrid')].copy()
base_test = base_test.dropna(subset=['RMSE'])
if not base_test.empty:
    best_base_name = base_test.loc[base_test['RMSE'].idxmin(), 'model']
    best_base_rmse = base_test['RMSE'].min()
else:
    best_base_name = None
    best_base_rmse = np.inf
hybrid_row = metrics_df[(metrics_df['phase'] == 'Test') & (metrics_df['model'] == 'Hybrid')]
hybrid_rmse = hybrid_row['RMSE'].values[0] if not hybrid_row.empty else np.inf
print(f"\nHybrid Test RMSE: {hybrid_rmse:.4f}")
if best_base_name:
    print(f"Best Base Model ({best_base_name}) Test RMSE: {best_base_rmse:.4f}")
else:
    print("No base model results available.")
if hybrid_rmse < best_base_rmse:
    print("Hybrid model outperforms all base models.")
else:
    print("Hybrid model did NOT outperform base models.")
plt.figure(figsize=(12,6))
model_names = metrics_df[metrics_df['phase']=='Test']['model'].unique()
rmse_vals = [metrics_df[(metrics_df['phase']=='Test') & (metrics_df['model']==m)]['RMSE'].values[0] for m in model_names]
sns.barplot(x=model_names, y=rmse_vals, palette='viridis')
plt.title('RMSE')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Error')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig(os.path.join(outd, 'rmse.png'), dpi=300)
plt.close()
npts = min(200, len(true_test))
steps = np.arange(npts)
plt.figure(figsize=(12,5))
plt.plot(steps, true_test[:npts], label='Actual', color='black')
plt.plot(steps, hybrid_test_preds[:npts], label='Hybrid', linestyle='--', color='red')
plt.legend()
plt.title('Hybrid Predictions')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(outd, 'hybrid_pred.png'), dpi=300)
plt.close()
for name, model in base.items():
    col = f'{name}_pred'
    if col in test_preds:
        plt.figure(figsize=(12,5))
        plt.plot(steps, true_test[:npts], label='Actual', color='black')
        plt.plot(steps, test_preds[col][:npts], label=name, linestyle='--')
        plt.legend()
        plt.title(f'{name} Predictions')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(outd, f'pred_{name}.png'), dpi=300)
        plt.close()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
pi = 0
for name in base.keys():
    col = f'{name}_pred'
    if col in test_preds:
        axes[pi].plot(steps, true_test[:npts], color='black', label='Actual')
        axes[pi].plot(steps, test_preds[col][:npts], '--', label=name)
        axes[pi].set_title(name)
        axes[pi].set_xlabel('Step')
        axes[pi].set_ylabel('Energy')
        axes[pi].legend()
        axes[pi].grid(alpha=0.3)
        pi += 1
axes[pi].plot(steps, true_test[:npts], color='black', label='Actual')
axes[pi].plot(steps, hybrid_test_preds[:npts], '--', label='Hybrid')
axes[pi].set_title('Hybrid')
axes[pi].set_xlabel('Step')
axes[pi].set_ylabel('Energy')
axes[pi].legend()
axes[pi].grid(alpha=0.3)
pi += 1
for idx in range(pi, len(axes)):
    axes[idx].axis('off')
plt.suptitle('All Models')
plt.tight_layout()
plt.savefig(os.path.join(outd, 'all_model.png'), dpi=300)
plt.close()
print("\nOK-End")