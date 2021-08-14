def agent2Dposition(x_info,agent_position):
    '''
    This function is to prepare the dataset that gets used to plot the animated visualisation of the results.
    This function puts all agents positions for each iteration and indicates the family that agent belongs to. 
    Parameters:
    ---------------
    x_info: This is the dataframe which contains each agents's characteristics including details about the family they belong to.
    agent_position: This is the dataframe that contains all the updated positions for each agent at different iterations.
    
    Returns:
    ---------------
    result : This is the dataframe with all individuals position details at different iterations including details about the family they belong to.
    '''
    import numpy as np
    import pandas as pd
    fam_id = x_info[['Family']]
    fam_id = fam_id.reset_index()
    
    df = agent_position.reset_index()
    iterations = np.unique(df['iter'])
    
    ind_pos = pd.DataFrame()
    for i in iterations:
        df2 = df[df['iter']==i]
        df2 = df2.drop(['iter'],axis=1)
        df2 = df2.set_index('position')
        df2 = np.transpose(df2)
        a=i.split('_')
        df2['iter'] = int(a[1])
        ind_pos = ind_pos.append(df2)
        df2.head()
    ind_pos = ind_pos.reset_index()
    ind_pos = ind_pos.sort_values(by=['iter'], ascending=True)
    result = pd.merge(ind_pos,fam_id, on='index')
    result = result.sort_values(by=['iter'], ascending=True)
    return result

def position2Dplot(result,lb,ub):
    ''' 
    This function uses the ploty package to plot the results.
    Parameters:
    -------------
    result:  This is the dataframe with all individuals position details at different iterations including details about the 
                family they belong to. Prepared using agent2Dposition((x_info,agent_position).
    
    Returns:
    ------------
    fig1 : This first figure shows the position of the agent within the whole community
    fig2 : This second figure shows the position of the agent within their respective families. 
    '''
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    pio.renderers.default = 'browser'
    fig = px.scatter(result, x="IndPosition0",y = "IndPosition1", 
           animation_frame="iter",
           animation_group="index",color="Family",hover_name="index",title='<br>Individual Position<br>',log_x=False, 
           range_x=[(lb-5),(ub+5)],range_y=[(lb-5),(ub+5)])
    
    fig2 = px.scatter(result, x="IndPosition0",y = "IndPosition1", 
           animation_frame="iter",
           animation_group="index",color="Family",hover_name="index", facet_col='Family',title='<br>Family Position Updates<br>',log_x=False, 
           range_x=[(lb-5),(ub+5)],range_y=[(lb-5),(ub+5)])

    return fig.show(),fig2.show()

def animated2Dvisualisation(df,agent_position,lb,ub):
    '''
    This function plots the animated plots showing how each agent moved around in the community and within their family in search of the optimal solution. 
    Parameters:
    ---------------
    x_info: This is the dataframe which contains each agents's characteristics including details about the family they belong to.
    agent_position: This is the dataframe that contains all the updated positions for each agent at different iterations.
    lb : Lower Bounds
    ub : Upper Bounds
    
    Returns:
    ------------
    fig1 : This first figure shows the position of the agent within the whole community
    fig2 : This second figure shows the position of the agent within their respective families. 
    '''
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    pio.renderers.default = 'browser'
    result = agent2Dposition(x_info,x_pos)
    fig1,fig2 = position2Dplot(result,lb,ub)
    return fig1,fig2
    
def convergenceRatePlot(ind_payoffs,objFunction):
    '''
    This function plots the convergence curve of the obejective function being investigated.
    Parameters:
    -------------
    ind_payoffs : These are the objective values of all agents in the community.
    objFunction : This is the objective function being solved for. 
    
    Returns:
    ------------
    fig : This is the convergence rate plot.
    '''
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    pio.renderers.default = 'browser'
    data = ind_payoffs.reset_index(drop=True)
    data['Objective Value'] = np.mean(-1*data,axis=1)
    data['Iteration'] = data.index
    fig = px.line(data, x='Iteration',y = "Objective Value",
              title=objFunction.__name__ + '<br>Convergence Rate<br>',
        line_shape="spline",render_mode="svg")
    
    return fig.show()