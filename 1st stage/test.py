import numpy as np
import matplotlib.pyplot as plt
pos=np.array([16.223297907011602,
13.932969717195553,
11.203566575394714,
9.1807832087317,
8.60800033352580,
9.968911527697987,
12.7582058375328,
16.14490388016829,
19.551823471515846,
22.497526329244234,
24.484728091348202,
25.083678144178734,
24.075646672008183,
21.47447545361494,
18.218139187778643,
15.462597391568215])
vel=np.array([16.149907027799895,
41.09879113045035,
77.1803147229792,
57.82376680748462,
38.17082126343763,
25.91232513506488,
16.860894658341522,
8.482053561732055,
1.675968436653447,
8.911846844425618,
19.186394133950213,
33.57653033458072,
54.33450990382952,
82.0008500604079,
65.3898603951579,
44.44790843584074,
30.00353399421032])
vel_x=np.array([i for i in range(len(vel))])
pos_x=np.array([i for i in range(len(pos))])

#设置绘图风格，使用科学论文常见的线条样式和颜色
plt.style.use('seaborn-whitegrid')
# 设置字体和字号
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }
plt.rc('font', **font)

# 绘制第一幅图像
plt.figure(1)
plt.plot(vel_x*3600, vel,color='blue', linewidth=1, label="Velocity-based impulse")
plt.plot(pos_x*3600, pos,color='red', linewidth=1, label="Position-based impulse")
plt.xlabel('time/s')
plt.ylabel('angle/°')
plt.title('Variation of Angles Between Impulse and Optimal Impulse')
plt.legend()
plt.tight_layout()
# 调整布局使得图像不溢出
plt.savefig('angle.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()