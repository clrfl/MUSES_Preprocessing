# We use Badges.xml of stackoverflow

#################################################################
# Convert xml to csv for memory-efficient preprocessing
#################################################################
with open('Badges.xml', 'r', encoding='utf-8') as f:
    with open('stackoverflow.csv', 'w', encoding='utf-8') as out:
        out.write('Id,UserId,Name,Date,Class,TagBased\n')
        for line in f.readlines():
            ln = line.strip()
            if ln.startswith('<row '):
                ln = ln[5:-3].split('"')
                ln = [ln[1], ln[3], ln[5], ln[7], ln[9], ln[11]]
                print(*ln, sep=',', file=out)

#################################################################
# Preprocessing
#################################################################
import pandas as pd

unique_badges = {'Autobiographer', 'Beta', 'Citizen Patrol', 'Civic Duty',
                 'Cleanup', 'Commentator', 'Critic', 'Editor', 'Woot!', 'Enthusiast',
                 'Fanatic', 'Generalist', 'Organizer', 'Scholar', 'Strunk & White',
                 'Student', 'Supporter', 'Teacher', 'Yearling', 'Sportsmanship', 'Promoter',
                 'Precognitive', 'Sheriff', 'Refiner', 'Tag Editor', 'Custodian',
                 'Suffrage', 'Analytical', 'Epic', 'Reviewer', 'Electorate', 'Explainer',
                 'Curious', 'Archaeologist', 'Vox Populi', 'Inquisitive', 'Convention', 'Deputy',
                 'Quorum', 'Excavator', 'Illuminator', 'Pundit', 'Investor', 'Talkative',
                 'Research Assistant', 'Legendary', 'Copy Editor', 'Altruist', 'Benefactor',
                 'Mortarboard', 'Synonymizer', 'Informed', 'Tenacious', 'Marshal', 'Unsung Hero',
                 'Proofreader', 'Constable', 'Outspoken', 'Disciplined', 'Peer Pressure', 'Taxonomist',
                 'Tumbleweed', 'Self-Learner'}
# yearling is semi-unique (max. 1x/year)
# reviewer and custodian are semi-unique (max 1x/type)
multiples_badges = {'Enlightened', 'Famous Question',
                    'Favorite Question', 'Good Answer', 'Good Question', 'Great Answer',
                    'Great Question', 'Guru', 'Necromancer', 'Nice Answer', 'Nice Question',
                    'Notable Question', 'Popular Question', 'Populist', 'Stellar Question', 'Lifejacket',
                    'Lifeboat', 'Booster', 'Steward', 'Publicist', 'Announcer', 'Socratic',
                    'Constituent', 'Caucus', 'Revival', 'Favorite Answer', 'Reversal'}

df = pd.read_csv("stackoverflow.csv")

# remove badges that cannot be awarded multiple times
for badge in unique_badges:
    df = df[df["Name"] != badge]

# remove tag based
df = df[df["TagBased"] == False]

#  we first select users who have earned at least 40 badges
user_badge_count = df['UserId'].value_counts()
valid_users = user_badge_count[user_badge_count >= 40].index
df = df[df['UserId'].isin(valid_users)]

# those badges which have been awarded at least 100 times
type_badge_count = df['Name'].value_counts()
valid_badges = type_badge_count[type_badge_count >= 100].index
df = df[df['Name'].isin(valid_badges)]

# remove the users who have been instantaneously awarded multiple
# badges due to technical issues of the servers.
df = df.groupby("UserId").filter(lambda x: x['Date'].is_unique)

# Find unclassified badges that remain in the dataset, but are not part of "multiples_badges"
#print("Unclassified badges:", set(df["Name"].unique().tolist()) - multiples_badges)

ts_collection = []
df = df.sort_values(by="Date")
for _, group in df.groupby('UserId'):
        group_df = group[["Date", "Name"]]
        ts_collection.append(group_df.T.to_json(orient="values"))

with open("full_preproc_stackexchange.jsonl", 'w', encoding='utf-8') as f:
    for s in ts_collection:
        f.write(s + '\n')


#################################################################
# Transform time series into target format
#################################################################
import pandas as pd
from datetime import datetime
import swifter
import json
from tqdm.auto import tqdm

badgelist = eval("{'Stellar Question', 'Strunk &amp; White', 'Favorite Question', 'Famous Question', 'Documentation Pioneer', 'Favorite Answer', 'Necromancer', 'Constituent', 'Great Question', 'Socratic', 'Guru', 'Great Answer', 'Documentation User', 'Nice Question', 'Reversal', 'Publicist', 'Notable Question', 'Good Question', 'Populist', 'Not a Robot', 'Announcer', 'Enlightened', 'Lifejacket', 'Nice Answer', 'Popular Question', 'Caucus', 'Census', 'Steward', 'Booster', 'Good Answer', 'Documentation Beta', 'Lifeboat', 'Revival'}")
badgedict = {}

for idx, x in enumerate(badgelist):
    badgedict[x] = idx

gesamt_df = pd.DataFrame(columns=['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])

results = []

with open('full_preproc_stackexchange.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=55000):
        df = pd.read_json(line).T
        df.columns = ['Date', 'Name']
        df['datetime'] = pd.to_datetime(df['Date'], format='%Y-%m-%dT%H:%M:%S.%f')
        df['type_event'] = df['Name'].apply(lambda y: badgedict[y])
        first_time = df['datetime'][0]
        df['time_since_start'] = (df['datetime'] - first_time).dt.total_seconds()/3600 # convert to hours
        df['time_since_last_event'] = (df['datetime'].diff().dt.total_seconds().fillna(0)) / 3600

        results.append([df['datetime'].to_list(), len(df), df['time_since_start'].to_list(), df['time_since_last_event'].to_list(), df['type_event'].to_list()])

    gesamt_df = pd.DataFrame(results, columns = ['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])
    gesamt_df.to_parquet('stack.parquet')

    gesamt_df['dim_process'] = len(badgelist)

    train = gesamt_df.sample(frac=0.8, random_state=42)
    train['seq_idx'] = range(len(train))
    train.to_parquet('train.parquet', index=False)
    remain = gesamt_df.drop(train.index)
    print(len(train), len(remain))

    val = remain.sample(frac=0.5, random_state=42)
    test = remain.drop(val.index)
    val['seq_idx'] = range(len(val))
    test['seq_idx'] = range(len(test))
    val.to_parquet('validation.parquet', index=False)
    test.to_parquet('test.parquet', index=False)
    print(len(val), len(test))
