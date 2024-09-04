# Deep Learning-Based Modeling for Power Converters via Physics-Guided Hierarchical Res-LSTM Frameworks
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-11.3.1-green.svg?style=plastic)
![MATLAB 2021b](https://img.shields.io/badge/MATLAB-2021b-blue.svg?style=plastic)
![PLECS 4.5.6](https://img.shields.io/badge/PLECS-4.5.6-green.svg?style=plastic)
[![Build status](https://ci.appveyor.com/api/projects/status/8msiklxfbhlnsmxp/branch/master?svg=true)](https://ci.appveyor.com/project/TadasBaltrusaitis/openface/branch/master)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

> [!Tip]
This is a GitHub repository for the **`Physics-Guided Hierarchical Res-LSTM Framework (PhyRes-LSTM)`**.

## Introduction 

Power converters are extensively utilized in many applications such as consumer electronics, industrial electronics, electric vehicles (EVs), energy storage systems, and distributed generation systems to generate either a regulated dc or ac voltage source and manage the power flow by mainly controlling the switching actions of power semiconductor devices (see Fig. 1)[[1]](https://ieeexplore.ieee.org/document/9351620). `In all of these applications, the ability to accurately describe the behavior of the converter is of critical importance, as it assists in component selection, controller design, and the preliminary evaluation of the entire DC system`.  Power converters inherently exhibit nonlinear behavior due to the switching actions of power switches, making their modeling challenging due to the complexity involved.

![1](https://github.com/user-attachments/assets/48175392-9e72-47fa-a75d-5fe416eeffe5)

Fig. 1. Diagram of a hierarchical microgrid with power converters [[2]](https://ieeexplore.ieee.org/document/9525187)

Generally, physical modeling relies on a knowledge of the values of the dc–dc converter parameters, both the passive elements and the control circuit. In addition, an accurate model should be able to reproduce ripples in currents and voltages, transients caused by switching on and off operations, etc., which is difficult to generate with physical-based modeling methods [[3]](https://ieeexplore.ieee.org/document/9492829). Over the past few years, classical machine learning (ML) and deep learning (DL) algorithms have been used for the data-driven modeling of power converters with simple topologies. It is worth noting that this modeling approach falls under the task of time series forecasting, so models that are related to time series and their theoretical analysis are also applicable in this situation. However, conventional time-series data-driven approaches for power converter modeling are **data-intensive, uninterpretable, and lack out-of-domain extrapolation capability**. Recent physics-informed modeling methods combine physics into data-driven models using loss functions, but they inherently suffer from `physical inconsistency, lower modeling accuracy, and require resource-intensive retraining for new case predictions.`

[^back to top](#top)
> [!Important]
In this paper, we propose a **`physics-guided hierarchical network with the deep residual network (ResNet) and long short-term memory (LSTM) for the data-driven modeling of power converters`**, which can bridge the connection between knowledge-based model and data-driven model for enhancing converter modeling and industrial process modeling. The main contributions of this work can be summarized as follow.

:triangular_flag_on_post:**(1)**  [A novel physics-guided framework based on DL is proposed](https://github.com/sub-p/PGRL), `improving the performance of converter modeling and industrial process modeling`.

:triangular_flag_on_post:**(2)** `Compared to the advanced existing physics-driven, data-driven, and physics-informed neural network (PINN)/ physics-guided machine learning (PGML) methods in the field of power electronics`, [our approach achieves superior physical consistency, data-light implementation, and generalization ability](https://github.com/sub-p/PGRL).

:triangular_flag_on_post:**(3)** **`This paper envisions to democratize artificial intelligence for the modeling of power electronic converters and systems. Meanwhile, this exemplary application envisions `**[providing a new perspective for tailoring existing machine learning tools for the power electronics field](https://github.com/sub-p/PGRL).

[^back to top](#top)

## Baselines

We will continuously add PINN/PGML models applied in the field of power electronics to expand this repository.

:white_check_mark: Numerical methods [[7]](https://ieeexplore.ieee.org/document/8409299)

:white_check_mark: LSTM [[8]](https://ieeexplore.ieee.org/document/9492829)

:white_check_mark: ResNet [[9]](https://arxiv.fropet.com/abs/1603.08029)

:white_check_mark: ResNet-LSTM [[10]](https://ieeexplore.ieee.org/document/9798792)

:white_check_mark: PINN [[11]](https://ieeexplore.ieee.org/document/9779551)

- [x] ...
      
[^back to top](#top)

## Requirements
- MATLAB == 2021b
- PLECS == 4.7.4
- Simulink
- Python 3.7
- torch == 1.9.0

[^back to top](#top)

## Reproducibility

> 1. **Install the requirement file**. For convenience, execute the following command.

```
pip install -r requirements.txt
```

[^back to top](#top)

> 2. **Jupyter Notebook Examples** : We provide Jupyter Notebook to help reproduce and customize our repo, which includes

```
1. PhyRes-LSTM-main.ipynb

2. Physical-driven_module_RK4.ipynb

3. DAB_TPS_waveform.ipynb
```

[^back to top](#top)

> 3. **Usage**:

3.1. Download the repository

3.2. Train the model.

```
run main.py
```

or

```
bash ./run.sh
```

[^back to top](#top)

## Problem Formulation
#### A. DAB Converters With Phase-Shift Modulation
Due to the advantages of galvanic isolation, high power densityand bidirectional energy flow, dual active bridge (DAB) dc–dc converter is gaining more and more popularity in industry applications. The topology of DAB dc–dc converter is depicted in Fig. 2. 

![image](https://github.com/user-attachments/assets/ff703b71-0dbc-4d9b-94dd-dfc2f000709b)

Fig. 2. Topology of dual-active-bridge DC–DC converter.

The most widely used control for DAB converters is the PSM from the invention of this topology. Consequently, single phase shift (SPS) control, extended phase shift (EPS) control, dual phase shift (DPS) control, and triple phase shift (TPS) control [[4]](https://ieeexplore.ieee.org/document/5776689) are proposed. Certainly, the TPS control can also be adapted to other methods such as SPS, DPS, etc., by simply modifying the phase shift ratio $D$. Therefore, we chose the more versatile TPS for our study. Based on the phase-shift ratio definition, TPS shows the highest control degree of freedom, which enables TPS the capability of providing the minimum root-mean-square (rms) current, lowest backflow power, and maximum ZVS range. 

![image](https://github.com/user-attachments/assets/cb6090d4-5c3a-43f9-b266-bcd67c51c780)

Fig. 3. Different modes and their typical waveforms of DAB converters with the triple-phase-shift control. (a) Mode 1:0 $\leq D_1 \leq D_2 \leq D_3 \leq 1$. (b) Mode 2:0 $\leq D_2 \leq D_1 \leq D_3 \leq 1$. (c) Mode 3: $0 \leq D_2 \leq D_3 \leq D_1 \leq 1$. (d) Mode 4: $0 \leq D_3-1 \leq D_1 \leq D_2 \leq 1$. (e) Mode 5: $0 \leq D_3-1 \leq D_2 \leq D_1 \leq 1$. (f) Mode 6: $0 \leq D_1 \leq D_3-1 \leq D_2 \leq 1$.

[^back to top](#top)

The related 6 cases are illustrated in Fig. 3 and their mathematical expressions are given as follows:

![image](https://github.com/user-attachments/assets/477b5414-ea28-45ec-9dd7-7f5f92a1b83c)

In our paper, we selected a typical and relatively more common mode as an example to illustrate our method. Of course, the approach is applicable to other models as well. 

![image](https://github.com/user-attachments/assets/2b4d2e56-4687-4b30-b5f7-436cc0017c1f)

Fig. 4. Simulation example of DAB converter.

[^back to top](#top)

#### B. Physics-Driven Modeling Approach for DAB Converters

A power electronics system can be expressed in the general form of partial differential equations (PDEs) through the state-space equations. To solve the state-space equation, numerical integration algorithms can be applied to discretize the continuous form. Formulation of integration can be written in the form:

![image](https://github.com/user-attachments/assets/4f6d0b0d-a0af-4a23-9a4f-103923c4d4de)

The exact solution of integration is analytic method. Integration computation can be done by using numerical integration methods such as: Euler, Heun, Runge Kutta or Adams-Bashforth-Moulton. Runge - Kutta integration method is the most popular method for PDE numerical solutions. This integration method has several advantages such as: ease of programming and stable. The fourth order Runge - Kutta integration method is:

[^back to top](#top)

#### *(1) Initialization*

- Initial Conditions: The initial values for the capacitor voltage $v_{C 2}(0)$ and the inductor current $i_L(0)$ at time $t=0$ are set as $v_{C 2,0}$ and $i_{L, 0}$ respectively.
- Time Step $h$ : A time step $h$ is chosen for the numerical integration. 
- Initial Time $t=0$ : The computation starts at $t=0$.

[^back to top](#top)

#### *(2) Computation of Intermediate Steps*

Calculate Intermediate Variables $k_1, k_2, k_3, k_4$ and $l_1, l_2, l_3, l_4$ :
- These variables are computed using the fourth-order Runge-Kutta method. They represent the approximate derivatives of the capacitor voltage and inductor current at various points within the current time step.
- The formulas involve parameters like $n D_2(t), R r, L_s, C_2$, and $R_{\text {load }}$, which correspond to circuit components such as the diode voltage drop, internal resistance, inductance, capacitance, and load resistance.
- Each $k$ and $l$ corresponds to the derivative approximations at different stages (current time point, midpoints, and a predicted next time point) within the time step.

![image](https://github.com/user-attachments/assets/637c5d39-9cd1-4b62-90af-42cc6110ec12)

[^back to top](#top)

#### *(3) Update* $v_{C 2}$ *and* $i_L$ *Using the Runge-Kutta Algorithm*

![image](https://github.com/user-attachments/assets/31f6d63a-c218-43a8-b1a5-8794acbe0c88)

[^back to top](#top)

#### C. Data-Driven Modeling Approach for DAB Converters
We first define the essence of the problem. The DAB converter’s modeling process can be considered a regression problem, since its goal is to obtain a function of the input–output mapping of the time series.

[^back to top](#top)

## Methodology
> [!Important]
The proposed PhyRes-LSTM is a novel PINN/PGML model designed to efficiently and accurately solve general nonlinear partial differential equations (PDEs) for the temporal behavior of power converters. Even in extreme low-data regime (data-light implementation), PhyRes-LSTM still demonstrates strong learning and generalization capabilities while strictly adhering to the integrated physical principles, addressing the challenges in the existing data-driven and state-of-the-art PINN methods in the power electronics field.

![Network structure visualization](https://github.com/user-attachments/assets/c302a36c-2de4-4798-9e85-267972158bc6)

Fig. 5. An overview of proposed PhyRes-LSTM approach

With the hierarchical ResNet-LSTM network and knowledge from prior information, it is possible for the proposed PhyRes-LSTM to generalize its performance with the limited number of training samples.

[^back to top](#top)
## Main Results
                                     
| **Model Ranking** |        **Model**               | **Challenges**      |
|----------|----------|----------|
|   🥇 1st     |**PhyRes-LSTM**                | Explainable and data-light; Extensibility and Robustness.| 
|   🥈 2nd     |**PGML/PGDL/PINN**             | Physical inconsistency; Loose constraints; Complete retraining.   | 
|   🥉 3rd     |**Purely data-driven modeling**| Heavy computation burden; Extensive datasets; Uninterpretable black-box models. |
|   🥉 4th     |**Knowledge-based modeling**     | The comprehensive theoretical knowledge is required; Limitations in accuracy and computational complexity.|

[^back to top](#top)

## Acknowledgement
> [!Tip]
This project is supported by the `National Natural Science Foundation of China under Grant 52007196`. At the same time, thank you all for your attention to this work! 

[^back to top](#top)

## Contact
> [!Note]
If you have any questions, feel free to contact Qianyi Shang [![image](https://github.com/user-attachments/assets/cc7d1dc3-951f-4635-9dd5-e2799c14f735)](https://orcid.org/0000-0002-4568-8895) through Email (21000504@nue.edu.cn) or Github issues. Pull requests are highly welcomed!

[^back to top](#top)
