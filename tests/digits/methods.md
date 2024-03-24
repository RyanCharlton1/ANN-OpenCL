Model result GrDsc 1e-1 learning rate: \
90/90   [====================>] loss: 0.17 batch time: 31 ms epoch time: 2.75s \
90/90   [====================>] loss: 0.07 batch time: 34 ms epoch time: 3.26s \
90/90   [====================>] loss: 0.05 batch time: 31 ms epoch time: 2.82s \
90/90   [====================>] loss: 0.04 batch time: 37 ms epoch time: 3.26s \
90/90   [====================>] loss: 0.03 batch time: 31 ms epoch time: 2.92s \
1000/1000[====================>] acc: 0.92 


Model result adam with 1e-3 learning rate: \
90/90   [====================>] loss: 0.07 avg. batch time: 27 ms epoch time: 2.49s \
90/90   [====================>] loss: 0.05 avg. batch time: 31 ms epoch time: 2.86s \
90/90   [====================>] loss: 0.04 avg. batch time: 28 ms epoch time: 2.53s \
1000/1000[====================>] acc: 0.93

Adam reaches similar performance in much less time, however using a higher learning rate resulted in gradient changes exploding the output.
