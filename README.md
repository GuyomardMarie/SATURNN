# Splines Approximation Through Understandable ReLU Neural Network (SATURNN)

This repository contains the algorithms for the SATURNN model.

**We provide here:**
- 'SATURNN_functions.ipynb': The different classes and functions in order to compute
    - SATURNN
    - Approximation by a Logistic Regression
- 'Main.ipynb':  An example of how to use the SATURNN 


## SATURNN

### Modeling

The SATURNN is a 1-Layer Neural Network for classification:
$$\Phi^\text{SATURNN}(x,\theta) = \sigma(\psi(x, \theta)),$$
with 
$\sigma$ the sigmoid activation:
$$\sigma(x) = \frac{1}{1+\exp(-x)},$$
and $\psi(x,\theta)$ the score function defined by :
$$\psi(x, \theta) = \frac{1}{\sqrt p} \left[ \beta_0 + \sum_{k=1}^p \beta_k \phi(s_kx_{\upsilon(k)} + b_k) \right],$$
with $\theta = [\beta^T, b^T]^T \in \mathbb{R}^{2p+1}$ are the trainable parameters and $x^T$ is the transpose of $x$ such that $x \in \mathcal{B}_2^d(0,r)$, $r>0$.

For the neuron $k$, we have:
- $\upsilon_k = \{1,\ldots,d\}$ is the input selector indicating which feature is handled by the neuron
- $b_k$ is the threshold from which the non-linear effect will be created
- $s_k = \{-1,1\}$ indicates if the non-linear effect will be created on the left or on the right of the treshold $b_k$
- $\beta_k$ indicates the impact of the non-linear effect on the estimated probability

### Initialization Process

For all $k \in \{1, \dots, p\}$:
- Fixed Parameters after initialization
    - $\upsilon_k(x) \sim \mathcal{U}[[ 1, d]]$
    - $s_k \sim \mathcal{B}(1/2)$
- Trainable parameters : $\theta^{(0)} = [\beta_0^{(0)}, \beta_1^{(0)}, \dots, \beta_p^{(0)}, b_1^{(0)}, \dots, b_p^{(0)}]$
    - $b_k \sim \mathcal{U}[-r, r]$
    - $\beta_k \sim \mathcal{N}(0,1)$
    
### Learning of the SATURNN

Learning SATURNN requires to minimize the following cost function:
$$\mathcal{L}^{\text{SATURNN}}(\theta)= \frac{1}{N} \sum_{i=1}^N L\left(\sigma(\psi(x^{(i)}, \theta)), y^{(i)}\right),$$
such that
$$\hat{\theta}^{\text{SATURNN}} = \arg\min_{\theta \in \mathcal{B}_2^{2p+1}(\theta^{(0)}, R)}   \mathcal{L}^{\text{SATURNN}}(\theta),$$
with $L(\cdot)$ the binary cross-entropy used for binary classification tasks 
$$L\left(\hat{y}, y\right)=-y\log(\hat{y})-(1-y)\log(1-\hat{y}).$$


![Schema_SATURNN](https://github.com/GuyomardMarie/SATURNN/assets/93378786/351ad06a-c9cb-4e33-9787-221e620fc15d)


## Approximation by a Logistic Regression

When the number $p$ of neurons composing the SATURNN is large enough, the SATURNN can be well approximated by:

$$\delta^\text{LR PSI LIN}(x, \eta) = g_0(x)^T\eta,$$

A Logistic Regression applied to the features pre-processed by the non-linear transformation $g_0(x) = \nabla_\theta \psi(x,\theta^{(0)})$ that is the gradient of the score function of the SATURNN $\psi(x,\theta)$ with respect to its parameters $\theta$ taken at the point $\theta^{(0)}$.

**Learning the LR PSI LIN**

Learning the LR PSI LIN requires to minimize the following cost function:
$$\mathcal{L}^{\text{LR PSI LIN}}(\theta)= \frac{1}{N} \sum_{i=1}^N L\left(\sigma(g_0(x^{(i)}, \eta)), y^{(i)}\right),$$
such that
$$\hat{\eta}^{\text{LR PSI LIN}} = \arg\min_{\eta \in \mathcal{B}_2^{2p+1}(\theta^{(0)}, R)}   \mathcal{L}^{\text{LR PSI LIN}}(\eta),$$
with $L(\cdot)$ the binary cross-entropy used for binary classification tasks.

**Equivalence with the SATURNN**

As establihed in Theorem 3.6 in [ICML, 2023]:

Let $\theta^{(0)} = [\beta_0^{(0)},\ldots,\beta_p^{(0)}, b_1^{(0)},\ldots,b_p^{(0)}]$ and $r, R>0$ such that $\beta_k^{(0)} \sim \mathcal{N}(0,1)$ and $b_k^{(0)} \sim \mathcal{U}[-r, +r]$. 
Then, we get
$$\sup_{\substack{\theta \in \mathcal{B}_2^{2p+1}(\theta^{(0)},R)\\ \eta \in \mathcal{B}_2^{2p+1}(0,R)}} \left|\mathcal{L}^\text{SATURNN}(\theta) - \mathcal{L}^{\text{LR}}(\eta) \right| \leq \frac{R^2}{2 \sqrt{p}}.$$


## Approximation by a Kernel Logistic Regression

Coming Soon...


## References

Article [1] : Marie Guyomard, Susana Barbosa, Lionel Fillatre, Kernel Logistic Regression Approximation of an Understandable ReLU Neural Network, International Conference on Machine Learning (ICML), 2023.

Article [2] : Marie Guyomard, Susana Barbosa, Lionel Fillatre, Understandable ReLU Neural Network for signal classification, International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023.

Article [3] (French) : Marie Guyomard, Susana Barbosa, Lionel Fillatre, R ́egression Logistique à Noyau équivalente à un Réseau de Neurones Interprétable, Groupe de Recherche et d’Etudes de Traitement du Signal et des Images (Gretsi), 2023.

Article [4] (French) : Marie Guyomard, Susana Barbosa, Lionel Fillatre, Approximation d’un Réseau de Neurones ReLU interprétable par une Régression Logistique à Noyau, Journées de la Statistique (JDS), 2023.

## Contacts 

Marie Guyomard : guyomard@i3s.unice.fr

Lionel Fillatre : lionel.fillatre@i3s.unice.fr

## Acknowledgements 
The method to visualize the partitionning of the input space is inspired by the work of Randall Balestriero provided here : https://github.com/RandallBalestriero/spline-DN-visualization
