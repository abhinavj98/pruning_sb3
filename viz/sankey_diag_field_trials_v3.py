import plotly.graph_objects as go

# Color palette
COLOR_PALETTE = {
    "Within Training Region": "#4C78A8", # green
    "Outside Training Region": "#F58518", # red
    "Success": "#54A24B",             # green (same as within training)
    "Failure": "#E45756",             # red (same as outside training)
    "Singularity": "#B279A2",         # purple
    "Outside Threshold": "#FF9DA6",    # pink-red
    "Collision": "#F28E2B",           # orange
    "Total UFO": "#FFB300",           # yellow
    #Different colors for Envy and Lab Envy
    "Total Envy": "#76B7B2",           # teal
    "Total Lab Envy": "#FFB6C1",       # light pink

}

class SankeyDiagramBuilder:
    def __init__(self, title="Sankey Diagram"):
        self.title = title
        self.nodes = []
        self.node_indices = {}
        self.links = []
        self.colors = []

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
        link_color = color if color else "gray"
        self.links.append(dict(source=src, target=tgt, value=value, color=link_color))

    def build(self):
        sources = [link["source"] for link in self.links]
        targets = [link["target"] for link in self.links]
        values = [link["value"] for link in self.links]
        link_colors = [link["color"] for link in self.links]

        # Assign x positions for node grouping (for nicer look)
        x_pos = {}
        for i, name in enumerate(self.nodes):
            if name.startswith("Type"):
                x_pos[i] = 0.0
            elif "UFO" in name and not "Total" in name:
                x_pos[i] = 0.1
            elif "Envy" in name and not "Total" in name and not "Lab" in name:
                x_pos[i] = 0.1
            elif "Lab Envy" in name and not "Total" in name:
                x_pos[i] = 0.1
            elif "Within Training Region" in name or "Outside Training Region" in name:
                x_pos[i] = 0.3
            elif "Total UFO" in name or "Total Envy" in name or "Total Lab Envy" in name:
                x_pos[i] = 0.5
            elif "Success" in name or "Failure" in name:
                x_pos[i] = 0.7
            else:
                x_pos[i] = 0.9
        x = [x_pos.get(i, 0.5) for i in range(len(self.nodes))]

        # Calculate total for percentages (sum of all top-level splits)
        total = 0
        for link in self.links:
            if "UFO" in self.nodes[link["source"]] or "Envy" in self.nodes[link["source"]] or "Lab Envy" in self.nodes[link["source"]]:
                total += link["value"]
        link_percents = [100 * v / total if total > 0 else 0 for v in values]

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            valueformat=".0f",
            node=dict(
                pad=30,
                thickness=30,
                line=dict(color="black", width=0.6),
                label=self.nodes,
                color=self.colors,
                x=x,
                hovertemplate='%{label}<extra></extra>'
                # No font property here!
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Count: %{value}<br>Percent: %{customdata:.1f}%<extra></extra>',
                customdata=link_percents
            )
        )])

        fig.update_layout(
            title_text=self.title,
            font=dict(size=18, color='black'),  # Set font globally
            margin=dict(l=30, r=30, t=80, b=30)
        )
        return fig

def add_percentages_to_labels(builder):
    # Sum values for each node as target (incoming flows)
    node_totals = [0] * len(builder.nodes)
    for link in builder.links:
        node_totals[link['target']] += link['value']
    total = sum(node_totals)
    # Append value and percent to label
    for i, name in enumerate(builder.nodes):
        if node_totals[i] > 0:
            percent = (node_totals[i] / total) * 100 if total > 0 else 0
            builder.nodes[i] = f"{name}<br><b>{node_totals[i]} ({percent:.1f}%)</b>"

# --- Build Sankey ---

builder = SankeyDiagramBuilder(title="Field Trials: UFO and Envy Results")

builder.add_link("UFO", "Within Training Region", 1, color=COLOR_PALETTE["Within Training Region"])
builder.add_link("UFO", "Outside Training Region", 13, color=COLOR_PALETTE["Outside Training Region"])
builder.add_link("Within Training Region", "Total UFO", 1, color=COLOR_PALETTE["Total UFO"])
builder.add_link("Outside Training Region", "Total UFO", 13, color=COLOR_PALETTE["Total UFO"])
builder.add_link("Total UFO", "Success", 5, color=COLOR_PALETTE["Success"])
builder.add_link("Total UFO", "Failure", 9, color=COLOR_PALETTE["Failure"])

builder.add_link("Envy", "Within Training Region", 10, color=COLOR_PALETTE["Within Training Region"])
builder.add_link("Envy", "Outside Training Region", 4, color=COLOR_PALETTE["Outside Training Region"])
builder.add_link("Within Training Region", "Total Envy", 10, color=COLOR_PALETTE["Total Envy"])
builder.add_link("Outside Training Region", "Total Envy", 4, color=COLOR_PALETTE["Total Envy"])
builder.add_link("Total Envy", "Success", 10, color=COLOR_PALETTE["Success"])
builder.add_link("Total Envy", "Failure", 4, color=COLOR_PALETTE["Failure"])

builder.add_link("Lab Envy", "Within Training Region", 10, color=COLOR_PALETTE["Within Training Region"])
builder.add_link("Lab Envy", "Outside Training Region", 0, color=COLOR_PALETTE["Outside Training Region"])
builder.add_link("Within Training Region", "Total Lab Envy", 10, color=COLOR_PALETTE["Total Lab Envy"])
builder.add_link("Total Lab Envy", "Success", 10, color=COLOR_PALETTE["Success"])

builder.add_link("Failure", "Singularity", 6, color=COLOR_PALETTE["Singularity"])
builder.add_link("Failure", "Outside Threshold", 6, color=COLOR_PALETTE["Outside Threshold"])
builder.add_link("Failure", "Collision", 1, color=COLOR_PALETTE["Collision"])


builder.nodes[builder.node_indices["UFO"]] = "UFO<br><b>14 ({:.2f}%)</b>".format(14/(14+14+10)*100)
builder.nodes[builder.node_indices["Envy"]] = "Envy<br><b>14 ({:.2f}%)</b>".format(14/(14+14+10)*100)
builder.nodes[builder.node_indices["Lab Envy"]] = "Lab Envy<br><b>10 ({:.2f}%)</b>".format(10/(14+14+10)*100)

builder.nodes[builder.node_indices["Within Training Region"]] = "Within Training Region<br><b>20 ({:.2f}%)</b>".format((0+10+10)/(14+14+10)*100)
builder.nodes[builder.node_indices["Outside Training Region"]] = "Outside Training Region<br><b>18 ({:.2f}%)</b>".format((14+4+0)/(14+14+10)*100)

builder.nodes[builder.node_indices["Total UFO"]] = "Total UFO<b><br>Success 5 ({:.2f}%) <br>Failure 9 ({:.2f}%)</b>".format(5/(5+9)*100, 9/(5+9)*100)
builder.nodes[builder.node_indices["Total Envy"]] = "Total Envy<b><br>Success 10 ({:.2f}%) <br>Failure 4 ({:.2f}%)</b>".format(10/(10+4)*100, 4/(10+4)*100)
builder.nodes[builder.node_indices["Total Lab Envy"]] = "Total Lab Envy<b><br>Success 10 ({:.2f}%)<br>Failure 0 ({:.2f}%)</b>".format(10/(10+0)*100, 0/(10+0)*100)

builder.nodes[builder.node_indices["Success"]] = "Success<br><b>25 ({:.2f}%)</b>".format((5+10+10)/(14+14+10)*100)
builder.nodes[builder.node_indices["Failure"]] = "Failure<br><b>13 ({:.2f}%)</b>".format((9+4+0)/(14+14+10)*100)

builder.nodes[builder.node_indices["Singularity"]] = "Singularity<br><b>6 ({:.2f}%)</b>".format(6/14*100)
builder.nodes[builder.node_indices["Outside Threshold"]] = "Outside Threshold<br><b>6 ({:.2f}%)</b>".format(6/14*100)
builder.nodes[builder.node_indices["Collision"]] = "Collision<br><b>1 ({:.2f}%)</b>".format(1/14*100)
fig = builder.build()
fig.show()


fig = builder.build()
fig.show()
