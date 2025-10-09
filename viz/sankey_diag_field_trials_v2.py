import plotly.graph_objects as go

# # Define nodes (can be named anything)
# labels = ["Type", "UFO",  "Tree0", "Tree1", "Tree2", "Tree3", "Within Training Region", "Outside Training Region"]
# #Make an enum for the labels
# class NodeLabels:
#     TYPE = 0
#     UFO = 1
#     TREE0 = 2
#     TREE1 = 3
#     TREE2 = 4
#     TREE3 = 5
#     WITHIN_TRAINING_REGION = 6
#     OUTSIDE_TRAINING_REGION = 7
# # Define flows
# # source and target are index-based (0 = "Input", 1 = "Process A", etc.)
# source = [0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
# target = [1, 2, 3, 4, 5, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8] # "Output" nodes
# values = [17 ,8, 4, 4, 1, 0, 8, 0, 4, 0, 4, 0, 1, 1, 1] # Number of points
#
# #Replace source and target with enum values
# source = [NodeLabels.TYPE, NodeLabels.UFO, NodeLabels.UFO, NodeLabels.UFO, NodeLabels.UFO,
#           NodeLabels.TREE0, NodeLabels.TREE0, NodeLabels.TREE1, NodeLabels.TREE1, NodeLabels.TREE2,
#           NodeLabels.TREE2, NodeLabels.TREE3, NodeLabels.TREE3, NodeLabels.WITHIN_TRAINING_REGION,
#           NodeLabels.WITHIN_TRAINING_REGION]
# target = [NodeLabels.UFO, NodeLabels.TREE0, NodeLabels.TREE1, NodeLabels.TREE2, NodeLabels.TREE3,
#             NodeLabels.WITHIN_TRAINING_REGION, NodeLabels.OUTSIDE_TRAINING_REGION, NodeLabels.WITHIN_TRAINING_REGION,
#             NodeLabels.OUTSIDE_TRAINING_REGION, NodeLabels.WITHIN_TRAINING_REGION, NodeLabels.OUTSIDE_TRAINING_REGION,
#             NodeLabels.WITHIN_TRAINING_REGION, NodeLabels.OUTSIDE_TRAINING_REGION, NodeLabels.WITHIN_TRAINING_REGION,
#             NodeLabels.OUTSIDE_TRAINING_REGION, NodeLabels.OUTSIDE_TRAINING_REGION]
#
#
# # Create Sankey diagram
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=20,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=labels,
#         color="blue"
#     ),
#     link=dict(
#         source=source,
#         target=target,
#         value=values,
#         color="rgba(150, 150, 250, 0.4)"  # semi-transparent flow colors
#     ))])
#
# fig.update_layout(title_text="Simple Sankey Diagram", font_size=12)
# fig.show()


import plotly.graph_objects as go

# class SankeyDiagramBuilder:
#     def __init__(self, title="Sankey Diagram"):
#         self.title = title
#         self.nodes = []
#         self.node_indices = {}
#         self.links = []
#         self.colors = []
#
#     def add_node(self, name, color="lightgray"):
#         if name not in self.node_indices:
#             self.node_indices[name] = len(self.nodes)
#             self.nodes.append(name)
#             self.colors.append(color)
#         return self.node_indices[name]
#
#     def add_link(self, source_name, target_name, value, color="gray"):
#         src = self.add_node(source_name)
#         tgt = self.add_node(target_name)
#         self.links.append(dict(source=src, target=tgt, value=value, color=color))
    #
    # def build(self):
    #     sources = [link["source"] for link in self.links]
    #     targets = [link["target"] for link in self.links]
    #     values = [link["value"] for link in self.links]
    #     link_colors = [link["color"] for link in self.links]
    #
    #     fig = go.Figure(data=[go.Sankey(
    #         node=dict(
    #             pad=20,
    #             thickness=20,
    #             line=dict(color="black", width=0.5),
    #             label=self.nodes,
    #             color=self.colors
    #         ),
    #         link=dict(
    #             source=sources,
    #             target=targets,
    #             value=values,
    #             color=link_colors
    #         ))])
    #
    #     fig.update_layout(title_text=self.title, font_size=12)
    #     return fig
import plotly.graph_objects as go

# Color palette
COLOR_PALETTE = {
    "UFO": "#4C78A8",
    "Envy": "#F58518",
    "Tree": "#A0CBE8",
    "Within Training Region": "#54A24B",
    "Outside Training Region": "#E45756",
    "Success": "#54A24B",
    "Failure": "#E45756",
    "Singularity": "#B279A2",
    "Outside Threshold": "#FF9DA6",
    "Collision": "#F28E2B",
}
import plotly.graph_objects as go

class SankeyDiagramBuilder:
    def __init__(self, title="Sankey Diagram"):
        self.title = title
        self.nodes = []
        self.node_indices = {}
        self.links = []
        self.colors = []

    # def auto_color(self, name):
    #     # Smart color assignment
    #     if "UFO" in name:
    #         return COLOR_PALETTE["UFO"]
    #     if "Envy" in name:
    #         return COLOR_PALETTE["Envy"]
    #     if "Within Training Region" in name:
    #         return COLOR_PALETTE["Within Training Region"]
    #     if "Outside Training Region" in name:
    #         return COLOR_PALETTE["Outside Training Region"]
    #     if "Tree" in name:
    #         return COLOR_PALETTE["Tree"]
    #     if "Success" in name:
    #         return COLOR_PALETTE["Success"]
    #     if "Failure" in name:
    #         return COLOR_PALETTE["Failure"]
    #     if "Singularity" in name:
    #         return COLOR_PALETTE["Singularity"]
    #     if "Outside Threshold" in name:
    #         return COLOR_PALETTE["Outside Threshold"]
    #     if "Collision" in name:
    #         return COLOR_PALETTE["Collision"]
    #     return "gray"

    def add_node(self, name, color=None):
        if name not in self.node_indices:
            self.node_indices[name] = len(self.nodes)
            self.nodes.append(name)
            assigned_color = color if color else "gray"
            self.colors.append(assigned_color)
        return self.node_indices[name]

    def add_link(self, source_name, target_name, value, color=None):
        src = self.add_node(source_name)
        tgt = self.add_node(target_name)
        link_color = color if color else self.auto_color(target_name)
        self.links.append(dict(source=src, target=tgt, value=value, color=link_color))

    def build(self):
        sources = [link["source"] for link in self.links]
        targets = [link["target"] for link in self.links]
        values = [link["value"] for link in self.links]
        link_colors = [link["color"] for link in self.links]

        # Assign x positions for nice ordering
        x_pos = {}
        for i, name in enumerate(self.nodes):
            if name in ["Type"]:
                x_pos[i] = 0.0
            elif "UFO" in name or "Envy" in name or "Lab Envy" in name:
                x_pos[i] = 0.1
            elif "Tree" in name and ("Within" not in name and "Outside" not in name):
                x_pos[i] = 0.3
            elif "Within Training Region" in name or "Outside Training Region" in name:
                x_pos[i] = 0.5
            elif "Success" in name or "Failure" in name:
                x_pos[i] = 0.7
            else:
                x_pos[i] = 0.9  # Reasons (Singularity, Outside Threshold, Collision)

        x = [x_pos.get(i, 0.5) for i in range(len(self.nodes))]
        #
        # fig = go.Figure(data=[go.Sankey(
        #     arrangement="snap",
        #     node=dict(
        #         pad=20,
        #         thickness=20,
        #         line=dict(color="black", width=0.5),
        #         label=self.nodes,
        #         color=self.colors,
        #         x=x
        #     ),
        #     link=dict(
        #         source=sources,
        #         target=targets,
        #         value=values,
        #         color=link_colors
        #     ))])
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            valueformat=".0f",  # No decimals
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=self.nodes,
                color=self.colors,
                x=x,
                hovertemplate='%{label}<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
            )
        )])

        fig.update_layout(title_text=self.title, font_size=20)
        return fig

COLOR_PALETTE = {
    "UFO": "#4C78A8",                 # soft blue
    "Envy": "#F58518",                # soft orange/pink
    "Tree": "#A0CBE8",                # light blue
    "Within Training Region": "#54A24B", # green
    "Outside Training Region": "#E45756", # red
    "Success": "#54A24B",             # green (same as within training)
    "Failure": "#E45756",             # red (same as outside training)
    "Singularity": "#B279A2",         # purple
    "Outside Threshold": "#FF9DA6",    # pink-red
    "Collision": "#F28E2B",           # orange
}

builder = SankeyDiagramBuilder(title="Field Trials: UFO and Envy Results")

builder.add_link("Type", "UFO", 14, color=COLOR_PALETTE["UFO"])
builder.add_link("UFO", "Within Training Region", 0, color=COLOR_PALETTE["Within Training Region"])
builder.add_link("UFO", "Outside Training Region", 14, color=COLOR_PALETTE["Outside Training Region"])
builder.add_link("Within Training Region", "Total UFO", 0, color=COLOR_PALETTE["Success"])
builder.add_link("Outside Training Region", "Total UFO", 14, color=COLOR_PALETTE["Failure"])
builder.add_link("Total UFO", "Success", 5, color=COLOR_PALETTE["Success"])
builder.add_link("Total UFO", "Failure", 9, color=COLOR_PALETTE["Failure"])

# Envy trees
builder.add_link("Type", "Envy", 14, color=COLOR_PALETTE["Envy"])
builder.add_link("Envy", "Within Training Region", 10, color=COLOR_PALETTE["Within Training Region"])
builder.add_link("Envy", "Outside Training Region", 4, color=COLOR_PALETTE["Outside Training Region"])
builder.add_link("Within Training Region", "Total Envy", 10, color=COLOR_PALETTE["Success"])
builder.add_link("Outside Training Region", "Total Envy", 4, color=COLOR_PALETTE["Failure"])
builder.add_link("Total Envy", "Success", 10, color=COLOR_PALETTE["Success"])
builder.add_link("Total Envy", "Failure", 4, color=COLOR_PALETTE["Failure"])

#Lab Envy
builder.add_link("Type", "Lab Envy", 10, color=COLOR_PALETTE["Envy"])
builder.add_link("Lab Envy", "Within Training Region", 10, color=COLOR_PALETTE["Within Training Region"])
builder.add_link("Lab Envy", "Outside Training Region", 0, color=COLOR_PALETTE["Outside Training Region"])
builder.add_link("Within Training Region", "Total Lab Envy", 10, color=COLOR_PALETTE["Success"])
builder.add_link("Total Lab Envy", "Success", 10, color=COLOR_PALETTE["Success"])

builder.add_link("Failure", "Singularity", 6, color=COLOR_PALETTE["Singularity"])
builder.add_link("Failure", "Outside Threshold", 6, color=COLOR_PALETTE["Outside Threshold"])
builder.add_link("Failure", "Collision", 1, color=COLOR_PALETTE["Collision"])

fig = builder.build()
fig.show()

