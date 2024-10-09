# ANN-OpenCL
A general purpose ANN framework made with C++ OpenCL

## Activation Functions

### (Leaky) ReLU

Zeros negative values, but unchanged otherwise. Leaky multiplies by some $\epsilon$ to stop neurons 'dying' when they hit zero and can't be changed anymore.

$$r(z_i)=\max(z_i, \epsilon z_i) = \begin{cases}
    z_i & \text{if} z_i > 0 \\ 
    \epsilon z_i & \text{otherwise}
\end{cases}$$

$$\frac{\partial r}{\partial z_i}=\begin{cases}
    1 & \text{if} z_i > 0 \\
    \epsilon & \text{otherwise}
\end{cases}$$

### Softmax 
Returns a vector summing to one, modelling probability

$\sigma(z_i) = \frac{e^{z_i}}{\Sigma_{j=0}^ne^{z_j}}$

Takes vector and returns a vector so differential is a matrix:

$$\frac{\partial\sigma}{\partial z} = 
\begin{pmatrix} \frac{\partial\sigma_1}{\partial z_1} & \frac{\partial\sigma_1}{\partial z_2} & \cdots & \frac{\partial\sigma_1}{\partial z_n} \\
\frac{\partial\sigma_2}{\partial z_1} & \frac{\partial\sigma_2}{\partial z_2} & \cdots\ & \frac{\partial\sigma_2}{\partial z_n} \\
\vdots & \vdots & \ddots & \vdots\\ 
\frac{\partial\sigma_n}{\partial z_1} & \frac{\partial\sigma_n}{\partial z_2} & \cdots & \frac{\partial\sigma_n}{\partial z_n}
\end{pmatrix}$$

By chain rule(valid as $\sigma_i$ will always be positive):

$\frac{\partial}{\partial z_j}\ln(\sigma_i) = \frac{1}{\sigma_i} \cdot \frac{\partial \sigma_i}{\partial z_j}$

$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i \cdot \frac{\partial}{\partial z_j}\ln(\sigma_i)$ 

$\ln(\sigma_i) = \ln(\frac{e^{z_i}}{\Sigma_{l=1}^ne^{z_l}}) = \ln(e^{z_i}) - \ln(\Sigma_{l=1}^ne^{z_l}) =z_i - \ln(\Sigma_{l=1}^ne^{z_l})$

$\frac{\partial}{\partial z_j}\ln(\sigma_i) = \frac{\partial z_i}{\partial z_j} - \frac{\partial}{\partial z_j}\ln(\Sigma_{l=1}^ne^{z_l})$
 
By chain rule:

$\frac{\partial}{\partial z_j}\ln(\Sigma_{l=1}^ne^{z_l}) = \frac{1}{\Sigma_{l=1}^ne^{z_l}}\cdot(\frac{\partial}{\partial z_j}\Sigma_{l=1}^ne^{z_l})$

$e^{z_j}$ is the only term in the sum effected by $z_j$:

$\frac{\partial}{\partial z_j}\Sigma_{l=1}^ne^{z_l} = e^{z_j}$

$\frac{\partial}{\partial z_j}\ln(\Sigma_{l=1}^ne^{z_l}) = \frac{e^{z_j}}{\Sigma_{l=1}^ne^{z_l}} = \sigma_j$

$$\frac{\partial z_i}{\partial z_j} = 
\begin{cases} 1 &\text{if } i = j \\
0 & \text{otherwise} 
\end{cases}$$

$\frac{\partial}{\partial z_j}\ln(\sigma_i) = \frac{\partial z_i}{\partial z_j} - \frac{\partial}{\partial z_j}\ln(\Sigma_{l=1}^ne^{z_l}) = 1\lbrace i=j \rbrace - \sigma_j$


$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i \cdot \frac{\partial}{\partial z_j}\ln(\sigma_i) = \sigma_i(1\lbrace i=j \rbrace - \sigma_j)$ 


$$\frac{\partial\sigma}{\partial z} = 
\begin{pmatrix} 
\sigma_1(1 - \sigma_1) & -\sigma_1\sigma_2 & \cdots & -\sigma_1\sigma_n \\
-\sigma_2\sigma_1& \sigma_2(1 - \sigma_2) & \cdots & -\sigma_2\sigma_n \\ 
\vdots & \vdots & \ddots &\vdots \\
-\sigma_n\sigma_1 & -\sigma_n\sigma_2 & \cdots & \sigma_n(1 - \sigma_n) 
\end{pmatrix}$$

## Loss Functions

### Mean Squared Error

$\text{MSE}= \frac{1}{2n}\sum(y-\hat{y})^2$

$\frac{\partial\mathcal{L}}{\partial\hat{y}}= \frac{\partial}{\partial \hat{y}} [\frac{1}{2n}\sum(y-\hat{y})^2]$

By chain rule:

$\frac{\partial\mathcal{L}}{\partial\hat{y}}-\frac{1}{n}\sum(y-\hat{y})$

### Categorical Cross Entropy

Compares information between two distributions. Needs a probability distribution as input so output layer would be softmax.

$\text{CE}= -\Sigma_{i=1}^ny_i\ln(\sigma_i)$

$\frac{\partial\mathcal{L}}{\partial z_j}=-\frac{\partial}{\partial z_j}\cdot\Sigma_{i=1}^ny_i\ln(\sigma_i)=-\Sigma_{i=1}^ny_i\cdot\frac{\partial}{\partial z_j}\ln(\sigma_i)=-\Sigma_{i=1}^n\frac{y_i}{\sigma_i}\cdot\frac{\partial\sigma_i}{\partial z_j}$

Using the softmax derivative:

$\frac{\partial\mathcal{L}}{\partial z_j}=-\Sigma_{i=1}^n\frac{y_i}{\sigma_i}\cdot\sigma_i\cdot(1\lbrace i=j \rbrace - \sigma_j)=-\Sigma_{i=1}^n y_i\cdot(1\lbrace i=j \rbrace - \sigma_j)$

$\frac{\partial\mathcal{L}}{\partial z_j}=\Sigma_{i=1}^n y_i \cdot \sigma_j - \Sigma_{i=1}^ny_i\lbrace i=j \rbrace$

$\Sigma_{i=1}^ny_i\lbrace i=j \rbrace = y_j$

$\frac{\partial\mathcal{L}}{\partial z_j}=\Sigma_{i=1}^n y_i \cdot \sigma_j - y_j$

Probability distributions sum to one so:

$\Sigma_{i=1}^ny_i=1$

$\frac{\partial\mathcal{L}}{\partial z_j}= \sigma_j - y_j$

$\frac{\partial\mathcal{L}}{\partial z}= \sigma - y$

## Optimisers

### Gradient Descent

Compute the loss gradient $g$ for the given parameter and move in the opposite direction multiplied by some learning rate $\alpha$.

$w_{t+1} = w_t - \alpha g_t$

### ADAM

Compute the exponential moving averages(parameterised by $\beta^1$ and $\beta^2$) for the loss gradient $g$, $m$, and its squared value $g^2$, $v$, these form a 'signal to noise' ratio. Using this the parameters can be updated less in sparser parameter spaces. The exponential moving averages are initialised at the zero vector, making initial estimates biased towards zero. To combat this introduce bias correction by dividing by $1-\beta^t$.

$m_t = \beta_1 m_{t-1} - (1-\beta_1)g$

$v_t = \beta_2 v_{t-1} - (1-\beta_2)g^2$

$\hat{m_t} = \frac{m_t}{1-\beta_1^t}$

$\hat{v_t} = \frac{v_t}{1-\beta_2^t}$

$w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$

For some very small $\epsilon$ to stop to stop the possibility of division by zero.

Reference: https://arxiv.org/pdf/1412.6980.pdf

## Regularisers

### Ridge Regression (L2)

Add the sum of squared weights to the loss to exponentially punish large weights preventing overfitting and gradient explosion.

$L = L + \frac{\lambda}{2}\Sigma_{i=1}^n w_i^2$

for a give weight $w_i$:

$\frac{\partial L_{reg}}{\partial w_i} = \frac{\partial L}{\partial w_i}+ \lambda w_i$

Lambda is a hyperparameter to control the severity regularisation. 

## Normalisation

Fitting the data to a normal distirbution(roughly) centering the mean at 0 and giving a vairance of 1. This removes problems of gradient magnitude like explosion and being too small to effectively change.

### Batch Normalisation

$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$

$y = \gamma \hat{x} + \beta$

The centering of the distribution around 0 makes pre normalisation bias redundant.

1DNorm normalises each feature across all samples in a batch.
2DNorm normalises all the pixels across all samples.
Each channel has its own $\beta$ and $\gamma$.

### Derivative

$\mu = \frac{1}{N}\sum_i^N x_i$

$\frac{\partial\mu}{\partial x_i} = \frac{1}{N}$

$\sigma^2 = E[X^2] - \mu^2 = \frac{1}{N}\sum_j^N x_j^2 - \mu^2$

$\frac{\partial\sigma^2}{\partial x_i} = \frac{\partial}{\partial x_i}[\frac{1}{N}\sum_j^N x_j^2 - \mu^2] = \frac{1}{N}\sum_j^N \frac{\partial}{\partial x_i}[x_j^2] - \frac{\partial}{\partial x_i}[\mu^2] = \frac{2}{N}x_i - \frac{2}{N}\mu = \frac{2}{N}(x_i - \mu)$

$\frac{\partial N}{\partial x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

$u = x_i - \mu$ 

$\frac{\partial u}{\partial x_i}=1 - \frac{1}{N}$

$v = (\sigma^2 + \epsilon)^{\frac{1}{2}}$

$\frac{\partial v}{\partial x_i} = \frac{1}{2}\frac{\partial}{\partial x_i}[\sigma^2 + \epsilon](\sigma^2 + \epsilon)^{-\frac{1}{2}} = \frac{x_i - \mu}{n\sqrt{\sigma^2 + \epsilon}}$

By qoutient rule:

$\frac{\partial \hat{x_i}}{\partial x_i} = \frac{(1-\frac{1}{n})(\sigma^2 + \epsilon)^{\frac{1}{2}} - \frac{(x_i - \mu)^2}{n\sqrt{\sigma^2 + \epsilon}}}{\sigma^2 + \epsilon} = \frac{(1-\frac{1}{N})(\sigma^2 + \epsilon) - \frac{1}{N}(x_i - \mu)^2}{(\sigma^2 + \epsilon)^{\frac{3}{2}}}$

