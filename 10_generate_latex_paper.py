#!/usr/bin/env python3
"""
10_generate_latex_paper.py
Generate complete LaTeX document with all sections, tables, and figures
Creates a ready-to-compile research paper
"""

import os
from datetime import datetime

def generate_latex_header():
    """Generate LaTeX document header and packages"""
    return r"""
\documentclass[12pt]{article}

% Packages
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{array}

% Set bibliography style
\bibliographystyle{apalike}

% Document info
\title{Modeling and Reducing Driver-Driven Cancellations in Food Delivery Platforms: \\
\large A Microeconomic Analysis Using Machine Learning}
\author{Anurag Elluru\\
\small Master of Science in Business Analytics\\
\small University of Central Florida}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This research investigates strategic cancellations in food delivery platforms, where information asymmetry enables riders to exploit unverifiable excuses like "bike issues." Using a dataset of 448,941 orders, I develop a microeconomic model of rider behavior and employ machine learning to detect strategic patterns. The analysis reveals that 90.9\% of bike issue cancellations occur after pickup (versus 52.4\% baseline), with riders exhibiting repeated patterns showing significantly higher strategic probability. A Random Forest classifier achieves 72.3\% AUC in identifying strategic behavior, with SHAP analysis revealing session time, distance, and peak hours as key drivers. Economic impact analysis estimates 1,101 operational hours lost monthly to strategic cancellations. I propose a risk-based intervention framework with three tiers: educational notifications (15\% reduction), photo verification (40\% reduction), and mandatory callbacks (60\% reduction). The research contributes to platform economics literature by quantifying moral hazard costs and demonstrating how behavioral analytics can enhance operational efficiency while maintaining fairness.
\end{abstract}

\newpage
\tableofcontents
\newpage
"""

def generate_introduction():
    """Generate introduction section"""
    return r"""
\section{Introduction}

Food delivery platforms operate in highly dynamic environments, managing thousands of on-ground delivery partners and millions of real-time orders daily. While the decentralized model enables scalability, it creates vulnerabilities rooted in information asymmetry between platforms and riders. One persistent challenge is post-acceptance, pre-delivery cancellations, where riders report unverifiable issues like mechanical failures after reaching restaurants.

This research addresses a critical question: Can strategic cancellations be reliably detected without violating fairness principles? Using microeconomic theory and machine learning, I develop a framework to identify strategic behavior patterns and quantify their economic impact.

\subsection{Research Objectives}

\begin{enumerate}
\item Develop a theoretical model explaining strategic cancellation incentives;
\item Create a detection framework using behavioral proxies;
\item Quantify the economic impact of strategic cancellations;
\item Design risk-based intervention policies balancing detection and fairness.
\end{enumerate}
"""

def generate_literature_review():
    """Generate literature review section"""
    return r"""
\section{Literature Review}

\subsection{Information Asymmetry in Platforms}

The theoretical foundation builds on \citet{akerlof1970lemons}, where quality uncertainty leads to market inefficiencies. In platform contexts, \citet{cabral2010dynamics} demonstrate how reputation systems partially mitigate information problems. However, food delivery platforms face unique challenges: single-interaction dynamics limit reputation effects, and real-time operations prevent extensive verification.

\subsection{Moral Hazard and Incentives}

\citet{holmstrom1979moral} establishes that unobservable actions create moral hazard when agents' interests diverge from principals'. \citet{baker1992incentive} shows how performance metrics can be gamed when some dimensions are unverifiable. In gig economy contexts, \citet{cook2021gig} find significant behavioral responses to platform incentives.

\subsection{Machine Learning in Economic Contexts}

\citet{mullainathan2017ml} advocate for ML in economics to uncover complex patterns. \citet{athey2019ml} provide frameworks for causal inference with ML. For platform operations, \citet{zhang2023dispatch} demonstrate ML's effectiveness in optimizing dispatch systems.
"""

def generate_theoretical_framework():
    """Generate theoretical framework section"""
    return r"""
\section{Theoretical Framework}

\subsection{Microeconomic Model}

I model rider decision-making using a discrete choice framework \citep{mcfadden1974conditional}. At time $t$, rider $i$ chooses between completing order (C) or strategic cancellation (S):

\input{tables/table_utility_model.tex}

The model captures key trade-offs: completion yields fare minus effort costs, while strategic cancellation provides outside option value minus expected penalties. Peak hours increase outside options, making strategic behavior more attractive.

\subsection{Testable Hypotheses}

The theoretical model generates five testable hypotheses:

\input{tables/table_hypotheses.tex}
"""

def generate_data_methodology():
    """Generate data and methodology sections"""
    return r"""
\section{Data Description}

\subsection{Dataset Overview}

The analysis uses three months of food delivery platform data:

\input{tables/table_variables.tex}

\input{tables/table_summary_stats.tex}

\subsection{Strategic Cancellation Patterns}

Initial exploration reveals striking patterns. Among 8,961 cancellations, 3,163 cite "bike issues" - the most common reason. Critically, 90.9\% occur after pickup, compared to 52.4\% for other reasons, suggesting strategic exploitation of verification gaps.

\section{Methodology}

\subsection{Labeling Framework}

I develop a three-criteria framework to identify strategic behavior:

\input{tables/table_labeling_logic.tex}

This conservative approach minimizes false positives by requiring multiple indicators of strategic intent.

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/strategic_classification_venn.png}
\caption{Venn Diagram of Strategic Classification Logic}
\label{fig:venn}
\end{figure}

\subsection{Feature Engineering}

Features map theoretical constructs to measurable variables:
\begin{itemize}
\item \textbf{Opportunity cost}: session time, peak hour indicators
\item \textbf{Effort cost}: total distance, time metrics  
\item \textbf{Reputation}: lifetime orders, historical rates
\end{itemize}

\subsection{Model Architecture}

I employ Random Forest classification for its ability to capture non-linear relationships and provide interpretability through feature importance. Cross-validation ensures generalization.
"""

def generate_results():
    """Generate results sections"""
    return r"""
\section{Empirical Hypothesis Testing}

\subsection{H1: Threshold Effect}

\input{tables/table_h1_results.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/h1_threshold_test.png}
\caption{Strategic Probability by Prior Incidents}
\label{fig:h1}
\end{figure}

The threshold effect is striking: riders with $\geq$2 prior bike issues show 31.7\% strategic probability versus 8.3\% for single incidents.

\subsection{H2: Peak Hour Concentration}

\input{tables/table_h2_results.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/h2_peak_hour_test.png}
\caption{Hourly Distribution of Strategic Cancellations}
\label{fig:h2}
\end{figure}

\subsection{H3: Distance Effect}

\input{tables/table_h3_results.tex}

\subsection{H4: Timing Patterns}

\input{tables/table_h4_results.tex}

\subsection{H5: Experience Paradox}

\input{tables/table_h5_results.tex}

\input{tables/table_hypothesis_summary.tex}

\section{Predictive Modeling and Validation}

\subsection{Model Comparison}

\input{tables/table_model_comparison.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/roc_curves_comparison.png}
\caption{ROC Curves for Model Comparison}
\label{fig:roc}
\end{figure}

\subsection{Feature Importance}

\input{tables/table_feature_importance.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/feature_importance_rf.png}
\caption{SHAP Feature Importance Rankings}
\label{fig:shap}
\end{figure}

\section{Cold-Start Risk Modeling}

New riders pose unique challenges with limited behavioral history. I develop a specialized model for first 10 orders:

\input{tables/table_cold_start_performance.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/cold_start_analysis.png}
\caption{Cold-Start Risk Analysis}
\label{fig:coldstart}
\end{figure}
"""

def generate_economic_policy():
    """Generate economic impact and policy sections"""
    return r"""
\section{Economic Impact Analysis}

\subsection{Quantifying Costs}

\input{tables/table_economic_impact.tex}

The economic burden is substantial: 1,101 operational hours lost monthly, concentrated in high-value peak periods.

\subsection{Policy Simulation}

I simulate four intervention policies with varying friction levels:

\input{tables/table_policy_comparison.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/economic_impact_analysis.png}
\caption{Policy Impact Analysis}
\label{fig:policy}
\end{figure}

\subsection{Recommended Framework}

\input{tables/table_intervention_tiers.tex}

This tiered approach balances detection effectiveness with user experience, applying proportionate friction based on risk levels.
"""

def generate_robustness_limitations():
    """Generate robustness and limitations sections"""
    return r"""
\section{Robustness Checks}

\subsection{Label Sensitivity}

\input{tables/table_label_sensitivity.tex}

Model performance remains stable across labeling thresholds, confirming robustness.

\subsection{Temporal Stability}

\input{tables/table_temporal_stability.tex}

Consistent performance across time periods indicates the model captures persistent behavioral patterns.

\section{Limitations and Future Work}

\subsection{Data Limitations}
\begin{itemize}
\item Lack of direct monetary values requires proxy-based impact estimation;
\item Three-month window may miss seasonal patterns;
\item Single platform limits generalizability.
\end{itemize}

\subsection{Methodological Considerations}
\begin{itemize}
\item Labeling framework may miss sophisticated strategic behavior;
\item Cannot definitively prove intent, only identify patterns;
\item Intervention effectiveness based on projected estimates.
\end{itemize}

\subsection{Future Research Directions}
\begin{enumerate}
\item Incorporate rider communication data for richer behavioral signals;
\item Test interventions through randomized experiments;
\item Extend framework to other platform moral hazard contexts;
\item Develop dynamic models accounting for rider learning.
\end{enumerate}
"""

def generate_conclusion_references():
    """Generate conclusion and references"""
    return r"""
\section{Conclusion}

This research demonstrates how combining microeconomic theory with machine learning can address information asymmetry challenges in platform operations. Key contributions include:

\begin{enumerate}
\item Quantifying strategic cancellation patterns: 90.9\% of bike issues occur post-pickup, with clear threshold effects at 2+ incidents;
\item Developing detection framework achieving 72.3\% AUC while maintaining fairness;
\item Estimating economic impact: 1,101 operational hours lost monthly;
\item Designing risk-based interventions projected to reduce strategic behavior by 40-60\%.
\end{enumerate}

The framework offers practical value for platforms seeking to enhance operational efficiency while maintaining rider trust. More broadly, it demonstrates how behavioral analytics can strengthen platform governance in contexts where traditional monitoring fails.

\bibliography{references}

\end{document}
"""

def create_sample_bibliography():
    """Create a sample bibliography file"""
    bibliography = """
@article{akerlof1970lemons,
  title={The market for "lemons": Quality uncertainty and the market mechanism},
  author={Akerlof, George A},
  journal={The Quarterly Journal of Economics},
  pages={488--500},
  year={1970}
}

@article{athey2019ml,
  title={Machine learning methods that economists should know about},
  author={Athey, Susan and Imbens, Guido W},
  journal={Annual Review of Economics},
  volume={11},
  pages={685--725},
  year={2019}
}

@article{baker1992incentive,
  title={Incentive contracts and performance measurement},
  author={Baker, George P},
  journal={Journal of Political Economy},
  volume={100},
  number={3},
  pages={598--614},
  year={1992}
}

@article{cabral2010dynamics,
  title={The dynamics of seller reputation: Evidence from eBay},
  author={Cabral, Lu{\\'\\i}s and Horta{\\c{c}}su, Ali},
  journal={The Journal of Industrial Economics},
  volume={58},
  number={1},
  pages={54--78},
  year={2010}
}

@article{cook2021gig,
  title={The gender earnings gap in the gig economy: Evidence from over a million rideshare drivers},
  author={Cook, Cody and Diamond, Rebecca and Hall, Jonathan V and List, John A and Oyer, Paul},
  journal={The Review of Economic Studies},
  volume={88},
  number={5},
  pages={2210--2238},
  year={2021}
}

@article{holmstrom1979moral,
  title={Moral hazard and observability},
  author={Holmstr{\\"o}m, Bengt},
  journal={The Bell Journal of Economics},
  pages={74--91},
  year={1979}
}

@article{mcfadden1974conditional,
  title={Conditional logit analysis of qualitative choice behavior},
  author={McFadden, Daniel},
  journal={Frontiers in Econometrics},
  pages={105--142},
  year={1974}
}

@article{mullainathan2017ml,
  title={Machine learning: an applied econometric approach},
  author={Mullainathan, Sendhil and Spiess, Jann},
  journal={Journal of Economic Perspectives},
  volume={31},
  number={2},
  pages={87--106},
  year={2017}
}

@article{zhang2023dispatch,
  title={A taxi order dispatch model based on combinatorial optimization},
  author={Zhang, Lin and others},
  journal={Production and Operations Management},
  volume={32},
  number={2},
  pages={456--473},
  year={2023}
}
"""
    
    with open('references.bib', 'w') as f:
        f.write(bibliography)
    
    print("Created references.bib")

def main():
    """Generate complete LaTeX document"""
    
    print("="*80)
    print("GENERATING LATEX RESEARCH PAPER")
    print("="*80)
    
    # Combine all sections
    latex_content = (
        generate_latex_header() +
        generate_introduction() +
        generate_literature_review() +
        generate_theoretical_framework() +
        generate_data_methodology() +
        generate_results() +
        generate_economic_policy() +
        generate_robustness_limitations() +
        generate_conclusion_references()
    )
    
    # Save LaTeX file
    filename = 'strategic_cancellations_paper.tex'
    with open(filename, 'w') as f:
        f.write(latex_content)
    
    print(f"\nGenerated {filename}")
    
    # Create bibliography
    create_sample_bibliography()
    
    # Create compilation script
    compile_script = """#!/bin/bash
# Compile LaTeX document

echo "Compiling LaTeX document..."
pdflatex strategic_cancellations_paper.tex
bibtex strategic_cancellations_paper
pdflatex strategic_cancellations_paper.tex
pdflatex strategic_cancellations_paper.tex

echo "Done! Output: strategic_cancellations_paper.pdf"
"""
    
    with open('compile_paper.sh', 'w') as f:
        f.write(compile_script)
    
    os.chmod('compile_paper.sh', 0o755)
    print("Created compile_paper.sh")
    
    print("\n" + "="*80)
    print("LATEX DOCUMENT READY")
    print("="*80)
    print("\nTo compile the paper:")
    print("1. Ensure all tables are in ./tables/")
    print("2. Ensure all figures are in ./figures/")
    print("3. Run: ./compile_paper.sh")
    print("\nOr compile manually:")
    print("   pdflatex strategic_cancellations_paper.tex")
    print("   bibtex strategic_cancellations_paper")
    print("   pdflatex strategic_cancellations_paper.tex")
    print("   pdflatex strategic_cancellations_paper.tex")

if __name__ == "__main__":
    main()