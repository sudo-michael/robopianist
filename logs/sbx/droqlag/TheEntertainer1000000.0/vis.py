from tbparse import SummaryReader
reader = SummaryReader('./events.out.tfevents.1691452877.cs-mars-04.3220211.0')
df = reader.scalars
print(df[df['tag'] == 'rollout/ep_rew_mean'])
