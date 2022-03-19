from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
from matplotlib import pyplot as plt

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    num = 1000
    X = np.random.normal(mu,sigma,num)
    uni_gauss = UnivariateGaussian()
    uni_gauss.fit(X)
    print(uni_gauss.mu_, uni_gauss.var_)

    # Question 2 - Empirically showing sample mean is consistent
    _, (ax1, ax2) = plt.subplots(2,constrained_layout=True,figsize=(8,8))
    diff = [abs(uni_gauss.fit(X[:i]).mu_ - mu) for i in range(10,num+10,10)]
    ax1.set(xlabel='Sample size',ylabel='Absolute distance')
    ax1.set_title('Distance between estimated expected value and actual expected value')
    ax1.scatter(range(10,num+10,10),diff,s=5)

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni_gauss.pdf(X)
    ax2.set(xlabel='Sample',ylabel='pdf')
    ax2.set_title('Empirical pdf')
    ax2.scatter(X,pdfs,s=7)
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    cov = np.array([[1,0.2,0,0.5],
                    [0.2,2,0,0],
                    [0,0,1,0],
                    [0.5,0,0,1]])
    X = np.random.multivariate_normal(mu,cov,1000)
    multi_gauss = MultivariateGaussian()
    multi_gauss.fit(X)
    print(multi_gauss.mu_)
    print(multi_gauss.cov_)

    # Question 5 - Likelihood evaluation
    # calculate heatmap
    _x, _y = np.linspace(-10,10,200), np.linspace(-10,10,200)
    x, y = np.meshgrid(_x,_y,sparse=True) # note that meshgrid reverses order- y is the first feature, x is the third
    func = np.vectorize(lambda u,v: multi_gauss.log_likelihood(np.array([u,0,v,0]),cov,X))
    heatmap = func(x,y)
    # plot heatmap
    plt.figure()
    plt.xlabel('First mean feature')
    plt.ylabel('Third mean feature')
    plt.title('Likelihood heatmap with respect to different first and third mean features')
    plt.pcolormesh(x, y, heatmap)
    plt.colorbar()
    plt.show()

    # Question 6 - Maximum likelihood
    xmax, ymax = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print(_y[ymax], _x[xmax])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
