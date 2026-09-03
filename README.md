## 🚨 AmbuRouteST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/vvchung/AmbuRouteST/blob/main/AmbuRouteST.ipynb)
[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Spatio‑Temporal Predictive Routing System for Ambulances, Integrating Dynamic Graph Modeling and Real‑Time Congestion Forecasting.

## 🎯 鳴笛開路，AI 搶救黃金 90 秒！

在 119 急救現場，「時間就是生命，分秒決定生死！」
當突發性心肌梗塞、重大車禍創傷發生時，救護車早一分鐘抵達，傷患就多一分存活的希望。然而，傳統導航（如 Google Maps）只看靜態距離或當下路況，當救護車開到半路時，突發的車禍回堵往往讓救護車卡在車陣中。
AmbuRouteST 是專為緊急救護打造的時空預測動態導航系統：利用時空圖卷積網路 (STGCN)，在車禍發生的瞬間，主動預測未來 15 分鐘內周邊街廓的「時空外溢壅塞效應」，並指導動態尋路演算法（Dynamic A*），指引救護車從最順暢、無阻礙的十字路口切入現場，幫急救任務節省寶貴的關鍵時間，對齊 119 勤務指揮中心的秒級調度實務。

------------------------------
## ⚡ 核心亮點功能 (Key Features)

* 🌐 實際路網動態建模： 整合 OSMnx 抓取真實城市拓撲結構，將真實的棋盤狀街道轉為圖論中的 Node 與 Edge。
* 🧠 STGCN 時空動態預測： 結合 GCN（圖卷積） 擷取路網空間依賴性，與 LSTM（長短期記憶） 預測時間序列流速。
* 🚑 消防救護實務校正： 拒絕不切實際的數學盲點！全面注入「救護車優先路權因子（鳴笛開路超速、逆向蠕行破障）」，確保 ETA（預估通行時間）符合 119 調度規範。
* 📊 戰情室級即時面板： 內建 HTML/CSS 戰情面板與動態路網壅塞熱度著色（Traffic Heatmap），一眼看穿塞車重災區！

------------------------------
## 🏗️ 技術架構與工作流 (System Architecture)

[ 119 報案注入 ] ──> [ OSMnx 載入高雄實際路網 ] 
                             │
                             ▼
                  [ 數據預處理: 張量重塑 ] (B, T, N, F_in)
                             │
                             ▼
         ┌──────────────────────────────────────┐
         │     STGCN 時空特徵擷取核心引擎         │
         │  1. GCN (Spatial): 矩陣相乘空間聚合    │
         │  2. LSTM (Temporal): 時間序列融合推論   │
         └──────────────────────────────────────┘
                             │
                             ▼
         [ 突發車禍回堵注入 / 救護路權車速校正 ]
                             │
                             ▼
         ┌──────────────────────────────────────┐
         │     動態尋路與視覺化渲染模組          │
         │  - 方案 A: 靜態最短路徑 (直衝重塞區)   │
         │  - 方案 B: Dynamic A* (時空智慧繞道)  │
         └──────────────────────────────────────┘
                             │
                             ▼
         [ Folium 戰情面板展示: 成功搶下黃金時間 ]

------------------------------
## 🔬 時空圖卷積 (STGCN) 原理解析說明
本系統的練習目標在於捕捉「空間（Spatial）」與「時間（Temporal）」的雙重依賴性。

## 1. 空間維度：圖卷積網路 (GCN) 訊息聚合
傳統卷積（CNN）只能處理規整的圖片像素，而城市路網是錯綜複雜的非歐幾里得圖結構（Non-Euclidean Graph）。我們透過以下核心公式進行空間特徵聚合：
$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W$$ 

* 鄰接矩陣 ($A$)：使用高斯核函數計算節點間的物理距離，距離越近，空間關聯權重越高。
* 度矩陣 ($D$)：進行拉普拉斯對角化對稱歸一化，防止節點因為連接道路過多而導致特徵爆炸。
* 特徵變換 ($W$)：利用矩陣相乘，讓每個十字路口（Node）在每個時間步都能完美融合周邊鄰近街道的動態車流資訊。

## 2. 時間維度：長短期記憶網路 (LSTM) 序列融合
圖卷積完成空間特徵聚合後，張量維度維持為 (B, T, N, F_hidden)。為了預測車流隨時間的演變，系統進行了張量重塑（Tensor Reshaping）：

## 將空間軸拼入 Batch 維度，轉為標準 LSTM 預期之三維結構
lstm_in = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, -1)

透過此步驟，將數據餵入 nn.LSTM，擷取過去 12 個時間步（例如過去 1 小時）的歷史車速趨勢，並在最終輸出層預測未來多步的車速變化。

## 3. 動態權重尋路：時變阻力 (Dynamic Cost) 計算
當 STGCN 預測出未來的時變車速 $V_{predicted}$ 後，系統將傳統的靜態長度權重更新為動態通行時間阻力（Dynamic Cost）：

$$\text{Dynamic Cost} = \frac{\text{Length}}{V_{predicted\_ms}}$$ 

當救護車行經前一個十字路口時，演算法偵測到前方路段的 $\text{Dynamic Cost}$ 因為車禍趨近無限大，便會主動觸發分流，規劃出綠色繞道方案！

------------------------------
## 📈 效果驗證 (Demo Evaluation)
透過物理總長度分攤與消防實務速限校正，系統在全球免費版 Google Colab (T4 GPU) 上模擬出了具實務說服力的成果：

* 方案 A (傳統直衝捷徑)：開鳴笛強行擠入車禍癱瘓區，總耗時 2 分 58 秒。
* 方案 B (AmbuRouteST 時空尋路)：提前感知前方癱瘓，走外圍大步繞道，總耗時 2 分 02 秒。
* 🎯 戰情成果：成功幫消防急救人員與危急傷患搶下「56 秒」的黃金救援時間！

------------------------------
## 🚀 如何使用 (How to Use)

點擊上方的 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gemini-generative-ai/colab-notebooks/blob/main/AmbuRouteST.ipynb) 按鈕，將專案內的 AmbuRouteST.ipynb 匯入 Google Colab，選擇 T4 GPU 核心，點擊「全部執行」，即可在 1 分鐘內渲染出位於高雄市新興區（中央公園周邊）的互動式救護戰情地圖！

------------------------------
Made with ❤️, ☕, and Gemini
