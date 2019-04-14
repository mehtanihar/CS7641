import networkx as nx
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def plot_nodes(dict_values, edge_values, title, policy=None,mode='markers+text'):
    values = []
    for key in dict_values:
        values.append(key)
    num_nodes  = len(values)
    G = nx.random_geometric_graph(num_nodes,0)
    # G = nx.path_graph(num_nodes)

    # pos = nx.spring_layout(G, k =0.15, iterations = 20)
    # nx.draw(G,pos)

    for i in range(num_nodes - 1):
        for j in range(i,num_nodes):
            if(values[j] in edge_values[values[i]]):
                G.add_edge(i,j)

    dmin = 1
    ncenter = 0
    # pos=nx.get_node_attributes(G,'pos')

    if(mode == 'markers+text'):
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            textposition='top center',
            mode=mode,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='Bluered',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=title,
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))
    else:
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            textposition='top center',
            mode=mode,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='Bluered',
                reversescale=True,
                color=[],
                size=5,
                colorbar=dict(
                    thickness=15,
                    title=title,
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))


    i = 0
    pos = nx.spring_layout(G, k =0.1, iterations = 20, seed = 1)
    # nx.draw(G,pos)
    for node in G.nodes():
        x, y = pos[i]#G.node[node]['pos']
        G.node[node]['pos'] = pos[i]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        if(mode == 'markers+text'):
            node_trace['text'] += tuple([str(values[i]) + ", Value: " + str(round(dict_values[values[i]],2))])
        node_trace['marker']['color']+=tuple([dict_values[values[i]]])
        i += 1
    

    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2 + (y - 0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')

    edges_pos = []
    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        if(mode == 'markers+text'):
            ratio = 0.3
        else:
            ratio = 0.1
        if(policy and values[edge[0]] in policy and policy[values[edge[0]]] == values[edge[1]]):
            edges_pos.append([x0,y0,x0+ratio*(x1-x0),y0+ratio*(y1-y0)])
        if(policy and values[edge[1]] in policy and policy[values[edge[1]]] == values[edge[0]]):
            edges_pos.append([x1,y1,x1+ratio*(x0-x1),y1+ratio*(y0-y1)])

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    autosize = True,
                    width = 1400,
                    title='<br>'+title,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=0,r=0,t=40),
                    annotations=[ 
                        dict(ax=edges_pos[i][0], ay=edges_pos[i][1], axref='x', ayref='y',
                        x=edges_pos[i][2], y=edges_pos[i][3], xref='x', yref='y') for i in range(0, len(edges_pos))
        
                        
                        # dict(
                        # text="",
                        # showarrow=False,
                        # xref="paper", yref="paper",
                        # x=0.005, y=-0.002 )
                         ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    #plotly.offline.iplot(fig, filename='networkx.html', image = "png", image_filename = title)
    plotly.offline.iplot(fig)