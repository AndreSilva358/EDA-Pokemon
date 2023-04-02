```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```


```python
pokemon_df = pd.read_csv(r'C:\Users\l0sky\OneDrive\Documentos\Pokemon.csv')
```


```python
print(pokemon_df.head())
```

       #                   Name Type 1  Type 2  Total  HP  Attack  Defense  \
    0  1              Bulbasaur  Grass  Poison    318  45      49       49   
    1  2                Ivysaur  Grass  Poison    405  60      62       63   
    2  3               Venusaur  Grass  Poison    525  80      82       83   
    3  3  VenusaurMega Venusaur  Grass  Poison    625  80     100      123   
    4  4             Charmander   Fire     NaN    309  39      52       43   
    
       Sp. Atk  Sp. Def  Speed  Generation  Legendary  
    0       65       65     45           1      False  
    1       80       80     60           1      False  
    2      100      100     80           1      False  
    3      122      120     80           1      False  
    4       60       50     65           1      False  
    

Aqui estamos importando os dados do dataset referentes aos 721 pokémons que existem da primeira até a sexta geração de Pokémon. Os pokémons tem um ID único, um nome, um ou dois tipos, e atributos: HP (health points), ataque, defesa, ataque especial, defesa especial, velocidade, geração, e lendário.


```python
print(pokemon_df.nunique())
```

    #             721
    Name          800
    Type 1         18
    Type 2         18
    Total         200
    HP             94
    Attack        111
    Defense       103
    Sp. Atk       105
    Sp. Def        92
    Speed         108
    Generation      6
    Legendary       2
    dtype: int64
    

Acima vemos o número de valores exclusivos para cada campo no conjunto de dados.

Considerando o número de valores únicos, nomes de colunas e meu conhecimento prévio sobre Pokémon, as colunas - 'Tipo 1', 'Tipo 2', 'Geração' e 'Lendário' podem ser consideradas como variáveis categóricas.


```python
print(pokemon_df['Type 1'].unique())
```

    ['Grass' 'Fire' 'Water' 'Bug' 'Normal' 'Poison' 'Electric' 'Ground'
     'Fairy' 'Fighting' 'Psychic' 'Rock' 'Ghost' 'Ice' 'Dragon' 'Dark' 'Steel'
     'Flying']
    


```python
print(pokemon_df['Type 2'].unique())
```

    ['Poison' nan 'Flying' 'Dragon' 'Ground' 'Fairy' 'Grass' 'Fighting'
     'Psychic' 'Steel' 'Ice' 'Rock' 'Dark' 'Water' 'Electric' 'Fire' 'Ghost'
     'Bug' 'Normal']
    


```python
print(pokemon_df['Generation'].unique())
```

    [1 2 3 4 5 6]
    


```python
print(pokemon_df['Legendary'].unique())
```

    [False  True]
    

Levando em consideração a exploração inicial, as seguintes variáveis podem ser consideradas como categóricas:

Tipo 1: Existem 18 tipos de Pokemon. Esse campo nos informa sobre o tipo ao qual um pokémon pertence.

Tipo 2: Alguns pokémons podem ter mais de um tipo. Esse campo nos informa sobre o segundo tipo de cada pokémon.

Geração: Pokémon possui várias gerações, e o dataset em questão tem dados sobre 6 deles. Esse campo nos diz a qual geração um Pokémon pertence.

Lendário: Existem alguns pokémons lendários na franquia que possuem poderes especiais em comparação com outros. Esse campo informa se um Pokémon é lendário ou não.


```python
df_numerical = pokemon_df.iloc[:, 5:11]
df_num_desc = df_numerical.describe()
print(df_num_desc)
```

                   HP      Attack     Defense     Sp. Atk     Sp. Def       Speed
    count  800.000000  800.000000  800.000000  800.000000  800.000000  800.000000
    mean    69.258750   79.001250   73.842500   72.820000   71.902500   68.277500
    std     25.534669   32.457366   31.183501   32.722294   27.828916   29.060474
    min      1.000000    5.000000    5.000000   10.000000   20.000000    5.000000
    25%     50.000000   55.000000   50.000000   49.750000   50.000000   45.000000
    50%     65.000000   75.000000   70.000000   65.000000   70.000000   65.000000
    75%     80.000000  100.000000   90.000000   95.000000   90.000000   90.000000
    max    255.000000  190.000000  230.000000  194.000000  230.000000  180.000000
    

Acima podemos ver os valores númericos do nosso dataset, os valores da média, desvio padrão, mínimo, percentil 25, percentil 50, percentil 75 e máximo.


```python
import sys
!{sys.executable} -m pip install scipy
```

    Requirement already satisfied: scipy in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (1.10.1)
    Requirement already satisfied: numpy<1.27.0,>=1.19.5 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from scipy) (1.24.2)
    


```python
df_numerical.plot.kde(figsize=(14,10))
```




    <Axes: ylabel='Density'>




    
![png](output_14_1.png)
    


Traçei um gráfico de kernel density plot para as variáveis numéricas, onde é possível ver a probabilidade da distribuição onde uma variável X pode assumir qualquer valor.


```python
labels = df_num_desc.columns
titles = df_num_desc.index
plt.style.use('ggplot')
fig, a = plt.subplots(4,2, figsize=(12,18))
c=0
for i in range(len(a)):
    for j in range(len(a[0])):
        y = df_num_desc.iloc[c]
        a[i, j].set_title(titles[c])
        a[i, j].bar(labels, y)
        c += 1

fig.suptitle("Aggregate stats vs Numerical Variables", fontdict={'weight':'bold'}, fontsize=20)
fig.tight_layout(pad=3)
plt.show()
```


    
![png](output_16_0.png)
    


No gráfico acima é possível inferir que todas as estatísticas agregadas são semelhantes para campos diferentes, com exceção do valor mínimo que varia muito.


```python
pokemon_df.isnull().sum()
```




    #               0
    Name            0
    Type 1          0
    Type 2        386
    Total           0
    HP              0
    Attack          0
    Defense         0
    Sp. Atk         0
    Sp. Def         0
    Speed           0
    Generation      0
    Legendary       0
    dtype: int64




```python
pokemon_df['Type 2'] = pokemon_df['Type 2'].fillna('NA')
```

Sabendo que muitos pokémons possuem apenas um tipo, decidi conferir os valores nulos presentes e identificá-los com 'NA' para que as informações ficassem mais claras.


```python
pokemon_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NA</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NA</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>109</td>
      <td>85</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>130</td>
      <td>85</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>159</td>
      <td>115</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7</td>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NA</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>50</td>
      <td>64</td>
      <td>43</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(pokemon_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 800 entries, 0 to 799
    Data columns (total 13 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   #           800 non-null    int64 
     1   Name        800 non-null    object
     2   Type 1      800 non-null    object
     3   Type 2      800 non-null    object
     4   Total       800 non-null    int64 
     5   HP          800 non-null    int64 
     6   Attack      800 non-null    int64 
     7   Defense     800 non-null    int64 
     8   Sp. Atk     800 non-null    int64 
     9   Sp. Def     800 non-null    int64 
     10  Speed       800 non-null    int64 
     11  Generation  800 non-null    int64 
     12  Legendary   800 non-null    bool  
    dtypes: bool(1), int64(9), object(3)
    memory usage: 75.9+ KB
    None
    

Vamos então analisar a relação entre diferentes variáveis.


```python
cor_numerical = df_numerical.corr()
print(cor_numerical)
```

                   HP    Attack   Defense   Sp. Atk   Sp. Def     Speed
    HP       1.000000  0.422386  0.239622  0.362380  0.378718  0.175952
    Attack   0.422386  1.000000  0.438687  0.396362  0.263990  0.381240
    Defense  0.239622  0.438687  1.000000  0.223549  0.510747  0.015227
    Sp. Atk  0.362380  0.396362  0.223549  1.000000  0.506121  0.473018
    Sp. Def  0.378718  0.263990  0.510747  0.506121  1.000000  0.259133
    Speed    0.175952  0.381240  0.015227  0.473018  0.259133  1.000000
    


```python
pip install seaborn 
```

    Requirement already satisfied: seaborn in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (0.12.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.17 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from seaborn) (1.24.2)
    Requirement already satisfied: pandas>=0.25 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from seaborn) (1.5.3)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from seaborn) (3.7.1)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.0.7)
    Requirement already satisfied: cycler>=0.10 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (4.39.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (23.0)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (9.5.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from pandas>=0.25->seaborn) (2023.3)
    Requirement already satisfied: six>=1.5 in c:\users\l0sky\appdata\local\programs\python\python311\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import seaborn as sns
```

Importei o seaborn e vou utilizá-lo para traçar o mapa de calor para a matriz de correlação acima encontrada usando o coeficiente de correlação de Pearson.


```python
sns.heatmap(cor_numerical, annot=True)
plt.show()
```


    
![png](output_28_0.png)
    


Ao olhar o mapa de calor acima é possível inferir que as variáveis numéricas na database não são altamente correlacionadas. Isso faz sentido, pois cada um desses atributos são pontuações de propriedades diferentes de um pokémon, e podem não ter grande correlação entre si.


```python
legendary_count_dist = pokemon_df['Legendary'].value_counts()
```


```python
labels = ['Non-Legendary', 'Legendary']
fig, a = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Legendary Distribution', fontdict={'weight':'bold'}, fontsize=20)

a[0].bar(labels, legendary_count_dist)
a[0].set_ylabel('No. of Pokemons')

a[1].pie(legendary_count_dist, labels=labels, explode=[0.05]*2, autopct='%.2f')

plt.show()
```


    
![png](output_31_0.png)
    


Acima, criei uma visualização traçando a contagem de pokémons lendários e não lendários.


```python
generation_count_dist = pokemon_df['Generation'].value_counts()
```


```python
labels = [i for i in range(1,7)]
fig, a = plt.subplots(1, 2, figsize=(15, 5))

a[0].bar(labels, generation_count_dist)
a[0].set_xlabel('Generation')
a[0].set_ylabel('No. of Pokemons')

a[1].pie(generation_count_dist, explode=[0.05]*len(generation_count_dist), autopct='%.2f', labels=labels)

fig.suptitle('Generation-wise Distribution', fontdict={'weight':'bold'}, fontsize=20)
plt.show()
```


    
![png](output_34_0.png)
    


Criei uma visualização traçando a contagem de pokémons por geração.


```python
dualpokemon_count_dist = [
    pokemon_df[pokemon_df['Type 2'] == 'NA']['#'].count(),
    pokemon_df[pokemon_df['Type 2'] != 'NA']['#'].count()
]
```


```python
labels = ['Non-Dual', 'Dual']

fig, a = plt.subplots(1, 2, figsize=(15, 5))

a[0].bar(labels, dualpokemon_count_dist)
a[0].set_ylabel('No. of Pokemons')

a[1].pie(dualpokemon_count_dist, explode=[0.02]*2, autopct='%.2f', labels=labels)

fig.suptitle('Dual Type Distribution', fontdict={'weight':'bold'}, fontsize=20)

plt.show()
```


    
![png](output_37_0.png)
    


Criei uma visualização traçando a contagem de pokémons de tipo único e duplo.


```python
type_count_dist = pokemon_df['Type 1'].value_counts()
```


```python
labels = list(type_count_dist.index)

fig, a = plt.subplots(2, 1, figsize=(15, 12))

a[0].bar(labels, type_count_dist)
a[0].set_xlabel('Type')
a[0].set_ylabel('No. of Pokemons')

a[1].pie(type_count_dist, labels=labels, explode=[0.03]*len(type_count_dist))

fig.suptitle('Type-wise Distribution', fontdict={'weight':'bold'}, fontsize=20)
plt.show()
```


    
![png](output_40_0.png)
    


Criei uma visualização da distribuição de tipos de pokémons. 


```python
pokemon_df['Overall'] = pokemon_df[['HP', 'Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def']].mean(axis=1)
```

Adicionei uma coluna ao dataset existente que pode nos dar uma ideia sobre o poder de um pokémon, onde ela será a média das variáveis numéricas: HP, Ataque, Defesa, Sp. Atk, Sp. Def e Speed.


```python
pokemon_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
      <td>53.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
      <td>67.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>87.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>104.166667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NA</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
      <td>51.500000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NA</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
      <td>67.500000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>109</td>
      <td>85</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
      <td>89.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>130</td>
      <td>85</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
      <td>105.666667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>159</td>
      <td>115</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
      <td>105.666667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7</td>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NA</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>50</td>
      <td>64</td>
      <td>43</td>
      <td>1</td>
      <td>False</td>
      <td>52.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
overalls_legendary = list(pokemon_df.groupby('Legendary')['Overall'])
```


```python
labels = ['Non-Legendary', 'Legendary']
overalls = [list(o[1]) for o in overalls_legendary]
colors = ['red', 'blue']

plt.figure(figsize = (10, 7))

plt.boxplot(overalls, labels = labels)

for i in range(len(labels)):
    x = []
    for j in overalls[i]:
        x.append(i+1)
    plt.scatter(x, overalls[i], c = colors[i])
    
plt.title('Legendary v/s Non-Legendary Pokemons', fontdict={'weight':'bold', 'fontsize':16})
plt.ylabel('Overall')
plt.show()
```


    
![png](output_46_0.png)
    


Comparei pokémons lendários com não lendários. É possível ver que os lendários são consideravelmente mais fortes que os não lendários.


```python
overalls_generation = list(pokemon_df.groupby('Generation')['Overall'])
```


```python
labels = [o[0] for o in overalls_generation]
overalls = [list(o[1]) for o in overalls_generation]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']

plt.figure(figsize = (10, 7))

plt.boxplot(overalls, labels = labels)

for i in range(len(labels)):
    x = []
    for j in overalls[i]:
        x.append(i+1)
    plt.scatter(x, overalls[i], c = colors[i])
    
plt.title('Generation-wise Pokemon Strength', fontdict={'weight':'bold', 'fontsize':16})
plt.ylabel('Overall')
plt.xlabel('Generation')
plt.show()
```


    
![png](output_49_0.png)
    


Comparei os pokémons por geração. É possível ver que a terceira geração (Hoenn) tem o maior valor geral de máximo, enquanto a quarte geração (Sinnoh) tem o valor mediano geral mais alto, e por fim a segunda geração (Johto) tem o valor mediano geral mais baixo.


```python
overalls_type = list(pokemon_df.groupby('Type 1')['Overall'])
```


```python
labels = [o[0] for o in overalls_type]
overalls = [list(o[1]) for o in overalls_type]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'olive']

plt.figure(figsize = (15, 7))

plt.boxplot(overalls, labels = labels)

for i in range(len(labels)):
    x = []
    for j in overalls[i]:
        x.append(i+1)
        
    if i < 9:
        plt.scatter(x, overalls[i], c = colors[i])
    else:
        plt.scatter(x, overalls[i], c = colors[i-9])
    
plt.title('Type-wise Pokemon Strength', fontdict={'weight':'bold', 'fontsize':16})
plt.ylabel('Overall')
plt.show()
```


    
![png](output_52_0.png)
    


Por fim, comparei os tipos de pokémons. É possível ver que os pokémons de tipo dragão são consideravelmente mais fortes que os demais, enquanto os pokémons tipo inseto são os mais fracos.
