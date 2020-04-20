from pattern import Circle, Checker
import matplotlib.pyplot as plt

c = Circle(100, 10, (50, 50))
c.draw()
plt.imshow(c.output)
plt.show()

c = Checker(100, 5)
c.draw()
plt.imshow(c.output)
plt.show()
