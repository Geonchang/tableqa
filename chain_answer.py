# %% 106
"""
Question: what is the last iec world plugs type in the list?
Ground Truth: ['N']
  IEC\nWorld Plugs\nType1               Standard Power\nrating Earthed  \
0                       A  NEMA 1-15 unpolarised    15 A/125 V      No   
1                       A    NEMA 1-15 polarised    15 A/125 V      No   
2                       A   JIS C 8303, Class II    15 A/100 V      No   
3                       B              NEMA 5-15    15 A/125 V    Yes*   
4                       B              NEMA 5-20    20 A/125 V    Yes*   

  Polarised Fused Insulated\npins Europlug\nCompatible  
0        No    No              No                   No  
1       Yes    No              No                   No  
2        No    No              No                   No  
3       Yes    No              No                   No  
4       Yes    No              No                   No  


문제
1. 컬럼명에 개행문자가 들어가서 그대로 사용 불가 \n > \\n
"""

df = one_sample(106).copy()
df.iloc[-1][df.columns[0]]
df.iloc[-1]["IEC\\nWorld Plugs\\nType1"]


# %% 409
"""
Question: when was the first win by decision?
Ground Truth: ['August 15, 2009']
   Res. Record        Opponent                Method  \
0   Win   12-3      Mike Hayes            KO (punch)   
1   Win   11–3  Nick Moghadden         TKO (punches)   
2  Loss   10–3   Guto Inocente  Decision (unanimous)   
3   Win   10–2     Brett Albee         TKO (strikes)   
4  Loss    9–2   Lavar Johnson          KO (punches)   

                                          Event                Date Round  \
0                  KSW 25: Khalidov vs. Sakurai    December 7, 2013     1   
1                                   Bellator 99  September 13, 2013     1   
2              Strikeforce: Barnett vs. Cormier        May 19, 2012     3   
3                   Strikeforce: Diaz vs. Daley       April 9, 2011     1   
4  Strikeforce Challengers: Bowling vs. Voelker    October 22, 2010     1   

   Time                              Location                    Notes  
0  1:12                       Wrocław, Poland                           
1  3:22   Temecula, California, United States           Bellator debut  
2  5:00   San Jose, California, United States  Light Heavyweight debut  
3  1:46  San Diego, California, United States                           
4  2:17     Fresno, California, United States 

문제
1. Column 정보만으론 Decision과 관련된 Column이 Method인걸 알기 어려움
2. 타입 변환, 문자열을 보고 data type을 판단해야 함
"""

df = one_sample(409).copy()
df = df[df["Method"].str.contains('Decision')]
df.loc[pd.to_datetime(df["Date"]).idxmin(), 'Date']

# %%
df

# %% 434
"""
Question: on which date did the home team member not score?
Ground Truth: ['4 March 2008']
               Date       Home team Score        Away team  \
0  17 December 2007          Foolad   4-4  Esteghlal Ahvaz   
1  21 December 2007        Rah Ahan   2-2       Steel Azin   
2  21 December 2007        Zob Ahan   1-1        Esteghlal   
3  21 December 2007    Bargh Shiraz   1-0    Paykan Tehran   
4  21 December 2007  Shahin Bushehr   2-2     Saba Battery   

                                      Notes  
0          Foolad progress 8-7 on penalties  
1        Rah Ahan progress 5-2 on penalties  
2       Esteghlal progress 3-1 on penalties  
3                                            
4  Shahin Bushehr progress 6-5 on penalties  

문제
1. score를 home/away team을 이해하고 parsing해야 함
   해당 사례에서는 home score away 컬럼순이었는데 반대라면?
"""

df = one_sample(434).copy()
df['Home team score'] = df['Score'].str.split('-').str[0].astype(int)
df.loc[df['Home team score'] == 0, 'Date'].values[0]


# %% 488
"""
Question: what was the name of the mission previous to cosmos 300?
Ground Truth: ['Luna 15']
         Launch date Operator               Name Sample origin  \
0      June 14, 1969           Luna E-8-5 No.402      The Moon   
1      July 13, 1969                     Luna 15      The Moon   
2  23 September 1969                  Cosmos 300      The Moon   
3    22 October 1969                  Cosmos 305      The Moon   
4    6 February 1970           Luna E-8-5 No.405      The Moon   

  Samples returned Recovery date                        Mission result  
0             None             -               Failure\nLaunch failure  
1             None             -     Failure\nCrash-landed on the Moon  
2             None             -  Failure\nFailed to leave Earth orbit  
3             None             -  Failure\nFailed to leave Earth orbit  
4             None             -               Failure\nLaunch failure  

문제
1. score를 home/away team을 이해하고 parsing해야 함
   해당 사례에서는 home score away 컬럼순이었는데 반대라면?
"""

df = one_sample(488).copy()
df.loc[df[df['Name'] == 'Cosmos 300'].index[0] - 1, 'Name']


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


# %% 8935
"""
Question: which country has produced the most drivers?
Ground Truth: ['France']
  Pos   Class  No                 Team  \
0  21  S\n1.5  42  Automobili O.S.C.A.   
1  22  S\n5.0   8          David Brown   
2  23  S\n1.1  63       Lucien Farnaud   
3  24  S\n750  49       Automobiles VP   
4  25  S\n5.0   5     Scuderia Ferrari   

                                     Drivers               Chassis  \
0         Jacques Péron\n Francesco Giardini    O.S.C.A. MT-4 1500   
1                Reg Parnell\n Roy Salvadori  Aston Martin DB3S SC   
2       Lucien Farnaud\n Adolfo Macchieraldo    O.S.C.A. MT-4 1100   
3  Yves Giraud-Cabantous\n Just-Emile Verney               VP 166R   
4               Robert Manzon\n Louis Rosier      Ferrari 375 Plus   

                              Engine Laps  
0                   O.S.C.A. 1.5L I4  243  
1  Aston Martin 2.9L Supercharged I6  222  
2                   O.S.C.A. 1.1L I4  199  
3                    Renault 0.7L I4  190  
4                   Ferrari 5.0L V12  177  


문제
1. (심각) Driver의 국적은 테이블 내에서 확인이 불가능...

해결 방안?
-> 드라이버를 검색해서 국적을 확인하는 에이전트?
-> 각 나라별 작명법에 따라 LLM이 국적을 판단?
"""

df = one_sample(8935).copy()
df


# %% 10476
"""
Question: what is the number of the only diesel locomotive from edwards rail car company?
Ground Truth: ['M-100']
  Number                   Builder        Type       Date Works number  \
0     51  Baldwin Locomotive Works   DS4-4-750       1949        74408   
1     52  Baldwin Locomotive Works   DS4-4-750       1949        74409   
2     53  Baldwin Locomotive Works  DS4-4-1000       1949        74193   
3     54  Baldwin Locomotive Works        S-12  1952/1953        75823   
4     55  Baldwin Locomotive Works       RS-12       1955        76024   

                                               Notes  
0  Acquired new 1949, Retired 1970/Wrecked-Scrapped.  
1  Acquired new 1949, retired in 1970 and scrappe...  
2  ex-Pan American Engineering W8380; née Army Co...  
3  ex-NW (3307); née WAB 307, wrecked 1968, retir...  
4  decorated for the United States Bicentennial\n...  


문제
1. 대문자 문제
2. idxmax로 접근할 경우 values를 안 써도 되는데, 필터로 접근하면 써야 함
"""

df = one_sample(10476).copy()
df.loc[df['Builder'].str.contains('Edwards Rail Car Company'), 'Number'].values[0]


# %% 11087
"""
Question: what symbol comes before symbol co?
Ground Truth: ['Fe']
  number symbol       name    21st    22nd    23rd    24th    25th 26th 27th  \
0     21     Sc   scandium  582163                                             
1     22     Ti   titanium  602930  639294                                     
2     23      V   vanadium  151440  661050  699144                             
3     24     Cr   chromium  157700  166090  721870  761733                     
4     25     Mn  manganese  158600  172500  181380  785450  827067             

  28th 29th 30th  
0                 
1                 
2                 
3                 
4                 


문제
1. chaining으로만 표현하면 너무 길어져 중간 변수 사용
"""

df = one_sample(11087).copy()
df['number_num'] = df['number'].astype(int)
var1 = df.loc[df['symbol'] == 'Co', 'number_num'].values[0]
df.loc[df['number_num'] == var1 - 1, 'symbol'].values[0]
# %%
