from pattern import Circle, Checker
from generator import ImageGenerator
import matplotlib.pyplot as plt

c = Circle(100, 10, (50, 50))
c.draw()
plt.imshow(c.output, cmap='gray')
plt.show()

c = Checker(100, 5)
c.draw()
plt.imshow(c.output, cmap='gray')
plt.show()

gen = ImageGenerator('exercise_data', 'Labels.json', 10, (64, 64, 3), True, True, True)

gen.show()
