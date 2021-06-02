import altair as alt

def plot_bar_confidence(source):

    plot = alt.Chart(source).mark_bar().encode(
        x="type",
        y="confidence",
        tooltip=['type', 'confidence'],
        color=alt.condition(
            alt.datum.confidence > 0,
            alt.value("steelblue"),  # The positive color
            alt.value("orange")  # The negative color
        )
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    ).properties(
        width=700, height=400
    ).interactive()

    return plot