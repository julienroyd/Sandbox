import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randint(80, 100, 100), columns=["score"])
df1["projectile"] = "baseball"
df2 = pd.DataFrame(np.random.randint(60, 70, 10), columns=["score"])
df2["projectile"] = "basketball"
df3 = pd.DataFrame(np.random.randint(60, 75, 20), columns=["score"])
df3["projectile"] = "freezbee"
df = pd.concat([df1, df2, df3])

# Calculate simple average
simple_average = df['score'].mean()
print(f"Average throw score? {simple_average:.2f}")

# Calculate multi-stage average
multi_stage_average = df.groupby('projectile').agg(['mean', 'count'])
print(f"How good is he in throwing sport in general? {multi_stage_average['score']['mean'].mean():.2f}")
multi_stage_average

