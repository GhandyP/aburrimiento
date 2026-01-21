# Create a flowchart using Plotly to show the Boredom Analysis System
import plotly.graph_objects as go
import plotly.io as pio

# Define node positions for the flowchart
nodes = {
    'A': {'pos': (0, 10), 'text': 'Usuario/Grupo', 'color': '#4A90E2'},
    'B': {'pos': (0, 8.5), 'text': 'Aplicación Flutter<br>Interfaz', 'color': '#4A90E2'},
    'C': {'pos': (0, 7), 'text': 'Ingreso 14 Indicadores<br>• Estructuras Sistémicas (4)<br>• Manifestaciones Grupales (4)<br>• Dimensiones Medición (6)', 'color': '#4A90E2'},
    'D': {'pos': (0, 5.5), 'text': 'Backend Python', 'color': '#34495E'},
    'E': {'pos': (-2, 4), 'text': 'Preprocesamiento', 'color': '#8E44AD'},
    'F': {'pos': (2, 4), 'text': 'Normalización', 'color': '#8E44AD'},
    'G': {'pos': (-2, 2.5), 'text': 'Random Forest<br>100 árboles', 'color': '#8E44AD'},
    'H': {'pos': (2, 2.5), 'text': 'Red Neuronal<br>4 capas', 'color': '#8E44AD'},
    'I': {'pos': (0, 1), 'text': 'Evaluación PyCM', 'color': '#27AE60'},
    'J': {'pos': (0, -0.5), 'text': 'Resultados<br>BAJO: 0.0-0.4<br>MEDIO: 0.4-0.7<br>ALTO: 0.7-1.0', 'color': '#27AE60'},
    'K': {'pos': (0, -2), 'text': 'Recomendaciones e<br>Intervenciones', 'color': '#27AE60'}
}

# Define connections between nodes
connections = [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('D', 'F'),
    ('E', 'G'), ('F', 'H'), ('G', 'I'), ('H', 'I'), ('I', 'J'), ('J', 'K')
]

# Create the figure
fig = go.Figure()

# Add arrows (connections)
for start, end in connections:
    x0, y0 = nodes[start]['pos']
    x1, y1 = nodes[end]['pos']
    
    # Add arrow line
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color='#333333', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add arrowhead
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='#333333',
        showarrow=True,
        text=''
    )

# Add nodes
for node_id, node_info in nodes.items():
    x, y = node_info['pos']
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(
            size=60 if len(node_info['text']) < 30 else 80,
            color=node_info['color'],
            line=dict(width=2, color='#333333')
        ),
        text=node_info['text'],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        showlegend=False,
        hoverinfo='text',
        hovertext=node_info['text']
    ))

# Update layout
fig.update_layout(
    title='Boredom Analysis System Architecture',
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-4, 4]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-3, 11]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False
)

# Save as both PNG and SVG
fig.write_image('boredom_analysis_flowchart.png')
fig.write_image('boredom_analysis_flowchart.svg', format='svg')

print("Flowchart created successfully using Plotly")