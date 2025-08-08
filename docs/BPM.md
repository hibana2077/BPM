好的 Tom，我把你的點子整理成一個可投 WACV 的「方法提案 + 實驗計畫」骨架，順手幫你把和 ultra-FGVC 的脈絡、對比/非對比自蒸餾的連結，以及「mean / higher-order 統計」延伸都串起來。下面直接給你一份**可拿去寫稿的結構化大綱**（含損失設計、實作細節與實驗建議）。文末也放上你上傳的 UFG 與 CLE-ViT 論文作為依據。

---

# 方法暫名

**BPM: Batch Prototype Matching for Ultra-FGVC**
（或取一個更有感的名字：**Mean-Diff Invariance (MDI)** / **ProtoSub**）

# 核心想法（你已經說得很清楚，我幫你公式化）

給定 batch $B=\{X_i,y_i\}_{i=1}^m$，定義「batch 級」原型：

$$
\bar{X}=\frac{1}{m}\sum_{i=1}^m X_i
$$

它抓到該 batch 的**共通背景/類內共性**。我們讓分類器 $f_\theta$ 同時看到 $X$、$\bar{X}$、以及「去共性」的 $X-\bar{X}$。訓練目標：

1. **分類不變性（像你說的）**

$$
\mathcal{L}_{\text{inv}}=\text{KL}\!\left(p_\theta(X)\,\Vert\,p_\theta(X-\bar{X})\right)\ \text{或}\ \lVert h_\theta(X)-h_\theta(X-\bar{X})\rVert_2
$$

其中 $p_\theta=\text{softmax}(f_\theta)$，$h_\theta$ 是倒數第二層 embedding。目的：扣掉 batch 共通成分後，判別仍穩定。

2. **原型不可分類（高熵/接近均勻）**

$$
\mathcal{L}_{\text{uni}}=\text{KL}\!\left(u\ \Vert\ p_\theta(\bar{X})\right)\quad(\text{或 }-\!H\!\left(p_\theta(\bar{X})\right))
$$

其中 $u$ 是均勻分佈。目的：$\bar{X}$ 只含「一般性」，不應帶可辨識線索。

3. **監督分類**

$$
\mathcal{L}_{\text{ce}}=\text{CE}\!\left(p_\theta(X),y\right)+\lambda_{\text{ce}}\ \text{CE}\!\left(p_\theta(X-\bar{X}),y\right)
$$

4. **非對比／雙網路自蒸餾（BYOL/SimSiam 型，無負樣本）整合**
   設 teacher $\phi$（EMA 參數），student $\theta$。送兩個「視圖」：

* 視圖 A：$X$ 的增強 $t_a(X)$ → teacher（**stop-grad**）
* 視圖 B：$(X-\bar{X})$ 的增強 $t_b(X-\bar{X})$ → student
  用 predictor $q_\theta$ 對齊：

$$
\mathcal{L}_{\text{sd}}=1-\cos\!\big(q_\theta(h_\theta(t_b(X-\bar{X}))),\ \text{sg}[h_\phi(t_a(X))]\big)
$$

再加上**center/variance** 正則避免崩塌（可用 VICReg/Barlow-style 的變異度/去相關正則，見下一節）。

**總損失：**

$$
\mathcal{L}=\mathcal{L}_{\text{ce}}+\alpha\,\mathcal{L}_{\text{inv}}+\beta\,\mathcal{L}_{\text{uni}}+\gamma\,\mathcal{L}_{\text{sd}}+\mathcal{R}_{\text{V\&C}}
$$

# Higher-order 統計的延伸（超越 mean）

把「batch 原型」從單一 $\bar{X}$ 提升為**低秩子空間**或**二階矩**：

* **子空間原型（PCA / learnable dictionary）**
  將 batch 影像視為長向量，取前 $k$ 個主成分 $\{P_j\}_{j=1}^k$（或學一組卷積型字典 $\mathcal{D}$），用投影 $\tilde{X}= \sum_j \alpha_j P_j$ 當「共享結構」。
  用 $X-\tilde{X}$ 取代 $X-\bar{X}$ 餵給模型，並讓 $\tilde{X}$ 同樣被壓向**不可分類**（高熵），鼓勵原型只吸收「非判別」部分。

* **特徵層二階對齊（VICReg/Barlow-style）**
  在 embedding space 對 pair $\big(h(X),h(X-\tilde{X})\big)$ 做**方差下限**與**去相關**正則：
  $\mathcal{R}_{\text{V\&C}}=\eta\cdot \text{VarPenalty}+\zeta\cdot \sum_{i\neq j}\text{Cov}_{ij}^2$。
  這等於把「一階（平均）」和「二階（協方差）」都納入「對去共性變換保持不變」的學習。

> 直觀地說：$\bar{X}$ 是 batch-wise **背景模板**；扣除它（或其子空間）後，剩下的是更「fine-grained 的差異」。Ultra-FGVC 的難點正是**小類間差異 / 大類內變異**，這種分解會把模型注意力拉向局部、細微線索（和既有 fine-grained 成果相呼應）。在 UFG 論文裡，他們也觀察到聚焦局部區域的方法（如 DCL/MaskCOV）於小樣本子集更有利。

# 與現有文獻的關係與新意

* **Ultra-FGVC 背景與挑戰**：UFG 基準有 47,114 張、3,526 類，並切成五個子集（SoyAgeing、SoyGene、SoyGlobal、SoyLocal、Cotton80），同時強調「大類內／小類間」挑戰。 &#x20;
* **局部/結構感知方法在 UFG 的優勢**：如 DCL、MaskCOV 在小樣本/小類間差異的子集更突出，支持我們把「共享成分抽走 → 強化微差異」的策略。
* **對比學習於 Ultra-FGVC**：CLE-ViT 證明了**實例級對比 + 自監督視圖（shuffle/mask）**能拉開類間、容忍類內變異，和我們「X 與 X−原型」視圖的精神一致；但我們主打的是**輸入空間的 batch 原型與二階統計**，並可**無負樣本**地用自蒸餾完成。&#x20;

> 補充：UFG 論文同時列了多種自監督/資料增強與二階特徵基線（SimCLR、MoCo v2、BYOL、fast-MPN-COV、DCL、MaskCOV…），提供你直接對比的參考表格與訓練規格。&#x20;

# 「mean 影像」在反傳後會變什麼樣？（生成觀點）

因為我們對 $\bar{X}$/$\tilde{X}$ 施加「高熵（不可分類）」壓力，反傳會把**具判別性的模式往平均/模糊方向沖淡**：

* 早期訓練：$\bar{X}$ 很像「模糊的平均圖」；
* 後期：若你用**子空間原型**，$\tilde{X}$ 會更像「低頻、共通結構」的**eigen-image**。
  建議在訓練中**可視化每個 epoch 的 $\bar{X}$/$\tilde{X}$**（或用 CAM 對比 $X$ 與 $X-\tilde{X}$ 的注意力差異），來展示「一般性 vs. 判別性」的分離；這和 UFG 論文裡用 CAM 去證明 DCL 更聚焦局部的做法一致。

# 具體訓練流程（簡潔版）

1. 取 batch，算 $\bar{X}$；可用**動量平均**（跨 batch 更穩）或**子空間原型**（PCA 或小型 conv-dict）。
2. 資料增強：對 $X$、$X-\bar{X}$ 各做一組增強（隨機裁切/顏色/旋轉等）。
3. 前向：同時輸入 $X$、$X-\bar{X}$、$\bar{X}$；teacher 看 $X$，student 看 $X-\bar{X}$。
4. 損失：$\mathcal{L}_{\text{ce}},\ \mathcal{L}_{\text{inv}},\ \mathcal{L}_{\text{uni}},\ \mathcal{L}_{\text{sd}},\ \mathcal{R}_{\text{V\&C}}$。
5. 反向：更新 student；teacher 做 EMA。
6. **higher-order**：若用 PCA 子空間，$k$ 建議 4–16；若用 learnable 原型，加**正交/稀疏**正則，避免吃進判別特徵。

# 與 BatchNorm 的差異（審稿常問）

BatchNorm 只做**通道均值/標準差**，不保留**空間結構**；我們的 $\bar{X}$/$\tilde{X}$ 是**具空間結構的 batch 原型**（甚至是多原型子空間）。目的不是穩定數值，而是**顯式剝離共享結構**，強化**微小差異**的可辨識度（這特別符合 ultra-FGVC 的小類間差異設定）。

# 實驗設計（直接對齊 UFG 基準）

* **資料與切分**：沿用 UFG 五子集與評估協定（包含每子集訓練/測試量；表格 2 規格可直接引用）。
* **對比方法**：復現官方表中的 13 個基線（AlexNet/VGG/ResNet、SimCLR/MoCo/BYOL、Cutout/CutMix、fast-MPN-COV、DCL、MaskCOV…）。
* **訓練細節**：影像 resize/crop、epoch、batch size、學習率排程等可直接沿用基準設定，以便公平比較。
* **主結果**：

  * 大樣本子集（SoyAgeing、SoyGene）與小樣本子集（SoyLocal、SoyGlobal、Cotton80）分開報；對齊 UFG 論文報法。
  * 期待小樣本/小類間差異的子集提升更明顯（和 DCL/MaskCOV 的觀察一致）。
* **消融**：

  1. 移除 $\mathcal{L}_{\text{inv}}$；2) 只用 $\bar{X}$，不用子空間；3) 只 supervised；4) 換**非對比**與**對比**兩種自監模組；5) $\beta$（高熵係數）掃參；6) batch size 影響（因為 $\bar{X}$ 品質依賴 batch 組成）。
* **可視化**：$\bar{X}/\tilde{X}$ 隨 epoch 的演化、$X$ vs $X-\tilde{X}$ 的 CAM 差異（呼應 UFG 的 CAM 分析）。
* **延伸到 transformer**：可參考 CLE-ViT 的視圖生成做法（shuffle/mask）混入我們的「原型視圖」，同時用實例級對比或無負樣本自蒸餾。

# 可能的審稿疑慮與回應

* **「這不就是把影像做 mean subtraction？」**
  不是。BN 是通道均值且無空間語義；我們是**空間對齊的 batch-wise 原型**與**高階子空間**，外加**不可分類約束**與**不變性學習**。
* **「batch 組成不穩會不會害你扣掉了判別特徵？」**

  1. 用動量原型（跨 batch 平滑）；2) 用多原型/子空間（低秩重建只吸收共性）；3) 以 $\mathcal{L}_{\text{inv}}$ 只在 embedding 階做對齊，避免輸入面誤差過度傳導。
* **「和現有 self-supervised 的差別？」**
  現有方法多在**特徵層**做不變性（不同增強視圖）；我們則在**輸入層顯式剝離共享結構**（$\bar{X}/\tilde{X}$），並把不變性/去相關規範**條件化在『去共性』操作上**。

# 為什麼這在 ultra-FGVC 特別合理？

UFG 的關鍵挑戰是**小類間差異 + 大類內變異**；本方法把 batch 共通模式吸走，讓模型自然「看」到**殘差裡的細節**。UFG 的基線也顯示：強化局部/結構與自監學習能帶來穩定提升（DCL/MaskCOV/CLE-ViT 等觀察）。&#x20;

---

## 實作參數建議

* $\lambda_{\text{ce}}=1$，$\alpha=0.5\sim1.0$，$\beta=0.1\sim0.5$，$\gamma=1$；VICReg/Barlow 的係數照論文預設起跳。
* 子空間階數 $k=8$（Cotton80/SoyLocal 可試 4/8/16）；動量原型 decay $m=0.9\sim0.99$。
* backbone：ResNet-50 與 Swin-B 各跑一套（CLE-ViT 用的 Swin-B 可以對齊）。

---

如果你要寫成 WACV 初稿，我可以幫你把上面的內容直接鋪成 **Abstract / Intro / Method / Experiments / Ablation / Visualization / Limitation / Broader Impact** 的分節稿，或先做一版 2-3 頁的 pre-submission proposal。要不要我直接開一份草稿幫你起筆？

---

### 參考（來自你上傳的論文）

* UFG（Ultra-FGVC 基準、五子集與挑戰）：資料規模、切分與「大類內/小類間」分析；基線與訓練細節可直接複用。  &#x20;
* DCL/MaskCOV 在小樣本子集的優勢（局部/結構感知）：
* CLE-ViT（實例級對比 + shuffle/mask 視圖，拉大類間距離、容忍類內變異）：&#x20;

> 註：我本來想幫你再補一些「無負樣本自蒸餾」（BYOL/SimSiam/DINO）與「去相關正則」（Barlow Twins/VICReg）的網路引用，不過我這邊目前抓不到外部頁面；先用你給的兩篇論文把實驗與動機站穩。要不要等你定案後，我再補全引用與 Related Work？
