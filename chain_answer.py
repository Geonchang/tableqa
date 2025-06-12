# %% 409
"""
Question: when was the first win by decision?
Ground Truth: ['August 15, 2009']
        Ship Name   Desig     Status  \
9   San Francisco   CA-38  Undamaged   
11        Detroit    CL-8  Undamaged   
12        Phoenix   CL-46  Undamaged   
16          Allen   DD-66  Undamaged   
17         Schley  DD-103  Undamaged   

                                                Notes Links  
9   Under overhaul at the Pearl Harbor Navy Yard b...  [10]  
11               Moored at berth F-13, aft of Raleigh  [12]  
12                                          Berth C-6  [13]  
16  Moored to Chew, Solace nearby to port, berth X-5.  [17]  
17  Moored in a nest of ships undergoing overhaul ...  [18]  

문제
1. Column 정보만으론 Decision과 관련된 Column이 Method인걸 알기 어려움
2. 타입 변환, 문자열을 보고 data type을 판단해야 함
"""

df = one_sample(409).copy()
df = df[df["Method"].str.contains('Decision')]
df.loc[pd.to_datetime(df["Date"]).idxmin(), 'Date']


# %% 1679
"""
Question: what is the first ship listed as undamaged?
Ground Truth: ['San Francisco']
      Ship Name  Desig                                             Status  \
0  Pennsylvania  BB-38                                            Damaged   
1       Arizona  BB-39                                   Sunk, total loss   
2        Nevada  BB-36       Seriously damaged, beached at Hospital Point   
3      Oklahoma  BB-37                                   Sunk, total loss   
4     Tennessee  BB-43  Relatively minor damage, repaired by February ...   

                                               Notes Links  
0  in drydock No. 1, with Cassin and Downes. Thre...   [1]  
1  Moored Battleship row, berth F-7 forward of Ne...   [2]  
2                 Moored aft of Arizona at berth F-8   [3]  
3  Moored Battleship row, outboard of Maryland at...   [4]  
4  Moored starboard side to berth F-6, next to We...   [5]  

문제
1. Column 정보만으론 undamaged과 관련된 Status이 Method인걸 알기 어려움
2. (사소한 문제) iloc가 쓰이는 경우 df가 반환되지 않으니 더 적절한 변수명이 필요할 듯
"""

df = one_sample(1679).copy()
df = df[df["Status"].str.contains("Undamaged")]
df.iloc[0]["Ship Name"]


# %% 1824
"""
Question: how many plants are in algeria?
Ground Truth: ['6']
    Plant Name    Location  Country Startup Date Capacity (mmtpa) Corporation
0  Qatargas II  Ras Laffan    Qatar         2009              7.8            
1   Arzew GL4Z              Algeria         1964             0.90            
2   Arzew GL1Z              Algeria         1978                             
3   Arzew GL1Z              Algeria         1997              7.9            
4  Skikda GL1K              Algeria         1972                             

문제
1. algeria가 데이터 상에선 대문자
"""

df = one_sample(1824).copy()
df = df[df["Country"] == "Algeria"]
len(df)


# %% 2286
"""
Question: which director has the most titles accredited to them?
Ground Truth: ['Patrick Norris']
   #                  Title     Directed by                     Written by  \
0  1                "Pilot"  Michael Dinner                   Peter Elkoff   
1  2                "Tessa"  Patrick Norris                    Liz Heldens   
2  3       "Surprise Party"  James Marshall  Kevin Falls & Matt McGuinness   
3  4        "Meteor Shower"      Craig Zisk  Kevin Falls & Matt McGuinness   
4  5  "My Boyfriend's Back"  David Straiton                   Dana Baratta   

  Original air date Production\ncode  
0     June 14, 2004           1AJE01  
1     June 21, 2004           1AJE02  
2     June 28, 2004           1AJE03  
3      July 5, 2004           1AJE04  
4     July 12, 2004           1AJE05  

문제
"""

df = one_sample(2286).copy()
df["Directed by"].value_counts().idxmax()


# %% 3657
"""
Question: who earned the most points at the 250cc valencian community motorcycle grand prix?
Ground Truth: ['Tohru Ukawa']
  Pos             Rider Manufacturer Time/Retired Points
0   1       Tohru Ukawa        Honda    49:50.449     25
1   2   Franco Battaini      Aprilia       +5.125     20
2   3   Loris Capirossi        Honda      +10.224     16
3   4     Shinya Nakano       Yamaha      +14.848     13
4   5  Stefano Perugini        Honda      +34.042     11

문제
"""

df = one_sample(3657).copy()
df.loc[pd.to_numeric(df['Points']).idxmax(), "Rider"]


# %% 4012
"""
Question: was the title for the ruler of the chinese vassal state lu king, marquis, or duke?
Ground Truth: ['Duke']
  State       Type       Name    Title Royal house    From      To
0   Chu  Sovereign       Huai     King          Mi  328 BC  299 BC
1   Han  Sovereign      Xiang  Marquis           —  311 BC  296 BC
2    Lu  Sovereign     Wen II     Duke          Ji  302 BC  280 BC
3    Qi  Sovereign        Min     King        Tian  300 BC  284 BC
4   Qin  Sovereign  Zhaoxiang     King        Ying  306 BC  251 BC

문제
1. 대문자 문제
2. idxmax로 접근할 경우 values를 안 써도 되는데, 필터로 접근하면 써야 함
"""

df = one_sample(4012).copy()
df.loc[df["State"] == "Lu", "Title"].values[0]


# %% 4506
"""
Question: which competition has the least notes?
Ground Truth: ['World Youth Championships']
   Year                    Competition               Venue  Position   Notes
0  2003      World Youth Championships  Sherbrooke, Canada       7th  1.75 m
1  2004     World Junior Championships     Grosseto, Italy       9th  1.80 m
2  2005  European Junior Championships   Kaunas, Lithuania       4th  1.82 m
3  2009  European Indoor Championships        Turin, Italy       5th  1.92 m
4  2010     World Indoor Championships         Doha, Qatar  10th (q)  1.89 m

문제
1. 1.75 m에서 1.75를 뽑아내야 함
"""

df = one_sample(4506).copy()
df['NoteValue'] = df['Notes'].str.extract(r'([0-9.]+)').astype(float)
df.loc[df['NoteValue'].idxmin(), 'Competition']