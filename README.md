# Self-organizing Maps (SOM)

This software implements a *Kohonen self-organizing map* in **Python** programming language for any clustering task.
This product is one of various for my thesis project. 

Feel free to use it anytime, and any reference to the original author is welcome! :sunglasses:

## Requirements
In order to use this software you must install the following dependencies:
* To execute **som.py**:
  * Python 3.4+
  * Numpy 1.13.3+

In order to verify the installation install this additional dependency or just to visualize the sample:
* matplotlib 3.0.3+

## Execution
In order to execute the sample software, first clone this repository and execute the software as:

```bash
git clone https://github.com/SkinMan95/self-organizing-maps.git
cd self-organizing-maps
python3 somgui.py
```

And it should start to iterate over an preestablished input in which it will output a unidimensional SOM in a separate window,
just as the following picture:

![SOM example](/images/som-example.png)

To execute the software for a different purpose you can checkout se code inside **som.py** to suit it to your needs, 
as it should be straightforward to modify if you are familiar with Python and just a little bit with 
[Numpy](https://www.numpy.org/), otherwise checkout [this](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf) 
cheatsheet or [this](https://cs231n.github.io/python-numpy-tutorial/) tutorial

## TODOs
- [ ] Implement command line interface to receive custom files and receive parameters from user :+1:
- [ ] Use *CuPy* library to push performance :rocket:

## Author
**Alejandro Anzola**, Computer Science student

Escuela Colombiana de Ingenieria Julio Garavito

Bogota, Colombia
