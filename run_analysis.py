"""
Quick start script to run the Neffy WoW vs Z-Score comparison analysis
"""

from scripts import main

# Run the complete analysis with data from data folder
fig, df_wow, df_zscore, differences = main(filepath='data/neffy_scripts_google_cloud.csv')

# Display the interactive visualization
fig.show()

# Optional: Save the figure as HTML
# fig.write_html("neffy_comparison_analysis.html")

# Optional: Export results to CSV
# df_wow.to_csv("neffy_wow_results.csv", index=False)
# df_zscore.to_csv("neffy_zscore_results.csv", index=False)
# differences.to_csv("neffy_method_disagreements.csv", index=False)

