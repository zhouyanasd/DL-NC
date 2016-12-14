import src

# define and generate the data
Data = src.data.Simple(5,400,3).Possion()

# define and initialize the liquid
Liquid = src.liquid.Liquid(Data, 'Input', 1)
Liquid.initialization()
Liquid.liquid_start()
Liquid.operate(1)




