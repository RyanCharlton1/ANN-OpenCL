# ANN-OpenCL
A general purpose ANN framework made with C++ OpenCL

## Activation Functions

### Softmax 
Returns a vector summing to one, modelling probability

$\sigma(z_i) = \frac{e^{z_i}}{\Sigma_{j=0}^ne^{z_j}}$

Takes vector and returns a vector so differential is a matrix:

$\frac{\partial\sigma}{\partial z} = 
\begin{pmatrix} \frac{\partial\sigma_1}{\partial z_1} & \frac{\partial\sigma_1}{\partial z_2} & \cdots & \frac{\partial\sigma_1}{\partial z_n} \\
\frac{\partial\sigma_2}{\partial z_1} & \frac{\partial\sigma_2}{\partial z_2} & \cdots\ & \frac{\partial\sigma_2}{\partial z_n} \\
\vdots & \vdots & \ddots & \vdots\\ 
\frac{\partial\sigma_n}{\partial z_1} & \frac{\partial\sigma_n}{\partial z_2} & \cdots & \frac{\partial\sigma_n}{\partial z_n}
\end{pmatrix}$ 

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

$\frac{\partial z_i}{\partial z_j} = \begin{cases} 1 &\text{if } i = j \\ 0 & \text{otherwise} \end{cases}$

$\frac{\partial}{\partial z_j}\ln(\sigma_i) = \frac{\partial z_i}{\partial z_j} - \frac{\partial}{\partial z_j}\ln(\Sigma_{l=1}^ne^{z_l}) = 1\lbrace i=j \rbrace - \sigma_j$


$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i \cdot \frac{\partial}{\partial z_j}\ln(\sigma_i) = \sigma_i(1\lbrace i=j \rbrace - \sigma_j)$ 


$\frac{\partial\sigma}{\partial z} = 
\begin{pmatrix} \sigma_1(1 - \sigma_1) & -\sigma_1\sigma_2 & \cdots & -\sigma_1\sigma_n \\  -\sigma_2\sigma_1& \sigma_2(1 - \sigma_2) & \cdots & -\sigma_2\sigma_n \\ \vdots & \vdots & \ddots &\vdots \\
-\sigma_n\sigma_1 & -\sigma_n\sigma_2 & \cdots & \sigma_n(1 - \sigma_n) \end{pmatrix}$

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

