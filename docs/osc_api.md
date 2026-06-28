# CoGaze OSC API

UDP 通信。デフォルトポートは設定変更可能。

| 方向 | ホスト | ポート | 既定値 |
|---|---|---|---|
| Python → Unity | Unity 側 IP | `--osc-port` | `9000` |
| Unity → Python | `localhost` | `--osc-recv-port` | `9001` |

---

## Unity → Python

### `/session/start [pid: str] [condition: str]`

セッションを開始する。Python 側のセッション設定ダイアログが開いていれば自動的に閉じて開始する。

| 引数 | 型 | 説明 |
|---|---|---|
| `pid` | string | 参加者ID |
| `condition` | string | `"IR"` / `"Webcam"` / `"WebcamFiltered"` / `"NoGaze"` のいずれか |

応答: `/experiment/ack ["session_start", "ok"]` または `["session_start", "error: session already active"]`

**注意:** セッションがすでにアクティブな場合は error を返し、新しいセッションは開始しない。  
事前に `/experiment/session_end` を送ってセッションを終了してから再送すること。

---

### `/calibration/start`

ウェブカメラキャリブレーション（16点ドットシーケンス）を開始する。  
Webcam / WebcamFiltered 条件のみ有効。完了後に `/calibration/result` が返る。

応答: `/experiment/ack ["calibration_start", "ok"]`

---

### `/calibration/abort`

キャリブレーションを中断する。`/calibration/result [0] [0.0] [0.0]` が返る。

応答: `/experiment/ack ["calibration_abort", "ok"]`

---

### `/experiment/trial_start [trial_id: str]`

trial を開始する。以降の CSV 行に `trial_id` が記録される。

| 引数 | 型 | 説明 |
|---|---|---|
| `trial_id` | string | 任意のトライアル識別子 |

応答: `/experiment/ack ["trial_start", "ok"]`

---

### `/experiment/trial_end`

trial を終了する。CSV の `trial_id` がクリアされる。

応答: `/experiment/ack ["trial_end", "ok"]`

---

### `/experiment/session_end`

セッションを終了し、Python 側をセッション設定画面に戻す。  
次のセッションは Python UI または `/session/start` で開始する。

応答: `/experiment/ack ["session_end", "ok"]`

---

### `/gaze/query`

最新のガズサンプルを即時返答させる（ポーリング用）。  
定常送信の `/gaze` で間に合う場合は不要。

応答: `/gaze` メッセージ（下記参照）

---

### `/ping`

疎通確認。

応答: `/pong`

---

## Python → Unity

### `/gaze x y mesh_certainty eye_certainty source condition`

ガズデータをリアルタイムで連続送信（キャリブレーション完了後、約 30 fps）。

| 引数 | 型 | 範囲 | 説明 |
|---|---|---|---|
| `x` | float | 0.0–1.0 | 正規化スクリーン X（左=0, 右=1） |
| `y` | float | 0.0–1.0 | 正規化スクリーン Y（上=0, 下=1） |
| `mesh_certainty` | float | 0.0–1.0 | MediaPipe ランドマーク品質 |
| `eye_certainty` | float | 0.0–1.0 | 目の開き度（ブレンドシェイプ由来） |
| `source` | string | — | `"ir"` / `"webcam"` / `"webcam:filtered"` 等 |
| `condition` | string | — | `"IR"` / `"Webcam"` / `"WebcamFiltered"` |

---

### `/face/metrics iod_norm face_cx face_cy status`

**Webcam / WebcamFiltered 条件のみ。** セッション中に約 30 fps で連続送信される。  
IR 条件では送信されない。  
Unity 側で顔位置ガイドを表示したり、キャリブレーション開始の自動ゲートとして使う。

| 引数 | 型 | 説明 |
|---|---|---|
| `iod_norm` | float | 現在の IOD（カメラフレーム幅に対する比率）。目標値は `config.py` の `IOD_TARGET_NORM`（既定 `0.10`、標準 640px カメラ・約 60 cm 想定） |
| `face_cx` | float | 顔中心 X（カメラ画像の正規化座標 0–1、左=0、右=1） |
| `face_cy` | float | 顔中心 Y（カメラ画像の正規化座標 0–1、上=0、下=1） |
| `status` | int | `0`=顔なし、`1`=遠すぎ、`2`=適切、`3`=近すぎ |

`face_cx` / `face_cy` はカメラ画像座標（スクリーン座標ではない）。  
`status=2` かつ `face_cx` ≈ 0.5、`face_cy` ≈ 0.5 のとき顔が正しく配置されている。

---

### `/calibration/started`

**Webcam / WebcamFiltered 条件のみ。** キャリブレーションウィンドウが開いた直後に一度だけ送信される。  
UI ボタンからの開始・OSC `/calibration/start` からの開始どちらでも送出される。  
Unity 側はこれを受けてキャリブレーション中フラグを立て、次フェーズへの遷移をブロックする。  
終了通知は `/calibration/result`（下記参照）。

引数なし。

---

### `/calibration/result quality err_x err_y`

キャリブレーション完了後に一度だけ送信される。

| 引数 | 型 | 説明 |
|---|---|---|
| `quality` | int | `2` = PASS, `1` = MARGINAL（閾値超過）, `0` = FAIL または中断 |
| `err_x` | float | 正規化スクリーン座標での X 方向 MAE（`-1.0` はサンプル不足で計算不能を意味する） |
| `err_y` | float | 正規化スクリーン座標での Y 方向 MAE（同上） |

品質閾値: `err_x <= 0.05` かつ `err_y <= 0.05` で PASS（`config.py` で変更可能）。  
MARGINAL は回帰フィット自体は成功しているが、精度が閾値を超えている状態。Unity 側でリトライを促すかどうか判断できる。

---

### `/experiment/ack command status`

Unity → Python コマンドへの応答。

| 引数 | 型 | 例 |
|---|---|---|
| `command` | string | `"session_start"`, `"trial_start"`, ... |
| `status` | string | `"ok"` または `"error: <message>"` |

---

### `/pong`

`/ping` への応答。引数なし。

---

## 条件別の送信メッセージ一覧

| 条件 | `/gaze` | `/face/metrics` | `/calibration/started` | `/calibration/result` |
|---|---|---|---|---|
| `IR` | ✓（キャリブ不要、即時） | — | — | — |
| `Webcam` | ✓（キャリブ後） | ✓ | ✓ | ✓ |
| `WebcamFiltered` | ✓（キャリブ後、One-Euro 済み） | ✓ | ✓ | ✓ |
| `NoGaze` | — | — | — | — |

`NoGaze` は IR ハードウェアが物理的に接続されているが、視線データを一切収集・送信しない。参加者には他の条件と区別がつかない。

---

## 典型的なシーケンス（Unity 主導）

```
Unity                               Python
  │                                   │
  │── /session/start P01 Webcam ────► │  セッション開始・CSV 記録開始
  │◄─ /experiment/ack session_start ──│
  │                                   │
  │  ── 顔位置調整フェーズ ──          │
  │◄─ /face/metrics 0.08 0.5 0.5 1 ──│  too far (status=1) → Unity でガイド表示
  │◄─ /face/metrics 0.10 0.5 0.5 2 ──│  good (status=2) → Unity で「OK」表示
  │                                   │
  │── /calibration/start ───────────► │  適切な位置確認後にキャリブ開始
  │◄─ /experiment/ack calibration_start│
  │◄─ /calibration/started ───────────│  ウィンドウ表示確認・Unity 側ブロック開始
  │     ［キャリブレーション実施中］    │
  │◄─ /calibration/result 2 0.03 0.04 │  完了通知（quality=2: PASS）→ ブロック解除
  │                                   │
  │── /experiment/trial_start T01 ──► │  trial 開始
  │◄─ /gaze 0.51 0.48 0.97 0.92 ir IR │  ガズ連続受信
  │── /experiment/trial_end ────────► │  trial 終了
  │                                   │
  │── /experiment/session_end ──────► │  セッション終了
```

---

## 接続設定例（Unity C#）

```csharp
// 受信（Python → Unity）
var receiver = new OSCReceiver(9000);

// 送信（Unity → Python）
var sender = new OSCSender("127.0.0.1", 9001);

// セッション開始
sender.Send(new OscMessage("/session/start", "P01", "Webcam"));

// 顔位置メトリクスを受け取る (Webcam/WebcamFiltered 条件のみ)
receiver.Bind("/face/metrics", (OscMessage msg) => {
    float iod_norm = (float)msg[0];  // IOD / frame_width
    float face_cx  = (float)msg[1];  // 0–1, camera image X
    float face_cy  = (float)msg[2];  // 0–1, camera image Y
    int   status   = (int)  msg[3];  // 0=no face, 1=too far, 2=good, 3=too close
});

// キャリブ開始を受け取る → Unity 側で次フェーズへの遷移をブロック
receiver.Bind("/calibration/started", (OscMessage msg) => {
    isCalibrating = true;  // 独自フラグ: 次フェーズ遷移を禁止する
});

// キャリブ結果を受け取る (quality: 2=PASS, 1=MARGINAL, 0=FAIL/aborted) → ブロック解除
receiver.Bind("/calibration/result", (OscMessage msg) => {
    isCalibrating = false;
    int quality  = (int)  msg[0];
    float err_x  = (float)msg[1];  // -1.0 = aborted with too few points
    float err_y  = (float)msg[2];
});
```
