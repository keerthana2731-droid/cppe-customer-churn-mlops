import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    
    # Title
    ax.text(7, 7.5, "Figure 4.1: Overall MLOps Platform Multi-Tier Architecture", 
            fontsize=18, fontweight='bold', ha='center', color='#0F172A', fontfamily='sans-serif')
    ax.text(7, 7.15, "Cloud Product and Platform Engineering - Customer Churn Prediction System", 
            fontsize=12, ha='center', color='#64748B', fontfamily='sans-serif')
    
    # Define Tiers / Boxes
    # [x, y, width, height, title, subtitle, items, bg_color, border_color]
    components = [
        # Tier 1: Jira
        (0.8, 4.3, 3.4, 2.3, "1. Agile Governance Tier", "Atlassian Jira Software", 
         ["• Backlog & Sprint Planning (Sprint 1)", "• User Stories: CHURN-1 to 4", "• Defect Tracking: CHURN-5 (Bug)", "• Workflow: To Do → In Progress → Done"],
         "#EFF6FF", "#2563EB"),
        
        # Tier 2: GitHub
        (5.3, 4.3, 3.4, 2.3, "2. Source Control Tier", "GitHub Repository (Git)", 
         ["• Repo: cppe-customer-churn-mlops", "• Branching: main (Protected)", "• Commit Auditing & Webhooks", "• .gitignore Artifact Protection"],
         "#F1F5F9", "#475569"),
        
        # Tier 3: Jenkins CI/CD
        (9.8, 4.3, 3.4, 2.3, "3. CI/CD Automation Tier", "Jenkins Automation Server", 
         ["• Declarative Pipeline-as-Code", "• Automated Multi-Stage Execution", "• PyTest Automated Gates", "• Local Host Port: 9090"],
         "#FEF3C7", "#D97706"),
        
        # Tier 4: MLflow
        (3.0, 1.0, 3.6, 2.4, "4. ML Governance Tier", "MLflow Tracking & Registry", 
         ["• Experiment: Customer_Churn_Prediction", "• Hyperparameters: n_estimators, max_depth", "• Metrics: Acc (86.5%), Prec, ROC-AUC", "• Cloudpickle Model Artifact Store"],
         "#ECFDF5", "#059669"),
         
        # Tier 5: Serving (FastAPI)
        (7.4, 1.0, 3.6, 2.4, "5. Production Serving Tier", "FastAPI Inference Engine", 
         ["• Uvicorn High-Throughput ASGI Server", "• Endpoints: /health, /predict", "• Pydantic Schema Validation", "• Real-Time Churn Scoring & Probabilities"],
         "#F5F3FF", "#7C3AED"),
    ]
    
    for (x, y, w, h, tier, tool, items, bg, border) in components:
        # Card shadow
        shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                        facecolor="#CBD5E1", edgecolor="none", alpha=0.3)
        ax.add_patch(shadow)
        # Card body
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(card)
        
        # Card Title
        ax.text(x + 0.15, y + h - 0.25, tier, fontsize=11, fontweight='bold', color=border)
        ax.text(x + 0.15, y + h - 0.55, tool, fontsize=12, fontweight='bold', color='#1E293B')
        
        # Separator line
        ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.65, y + h - 0.65], color=border, linewidth=1, alpha=0.5)
        
        # Bullet items
        curr_y = y + h - 0.95
        for item in items:
            ax.text(x + 0.15, curr_y, item, fontsize=9.5, color='#334155')
            curr_y -= 0.32

    # Draw connection arrows
    # Jira -> GitHub
    ax.annotate("", xy=(5.2, 5.45), xytext=(4.3, 5.45),
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=2.5, mutation_scale=15))
    ax.text(4.75, 5.65, "Task Link", fontsize=9, ha='center', color='#2563EB', fontweight='bold')
    
    # GitHub -> Jenkins
    ax.annotate("", xy=(9.7, 5.45), xytext=(8.8, 5.45),
                arrowprops=dict(arrowstyle="->", color="#D97706", lw=2.5, mutation_scale=15))
    ax.text(9.25, 5.65, "Git Push / Hook", fontsize=9, ha='center', color='#D97706', fontweight='bold')
    
    # Jenkins -> MLflow
    ax.annotate("", xy=(5.0, 3.5), xytext=(10.5, 4.2),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2.5, mutation_scale=15,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(7.6, 3.9, "Trigger Train & Log", fontsize=9.5, ha='center', color='#059669', fontweight='bold')
    
    # Jenkins -> FastAPI
    ax.annotate("", xy=(9.2, 3.5), xytext=(11.5, 4.2),
                arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=2.5, mutation_scale=15,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(10.7, 3.75, "Deploy / Smoke Test", fontsize=9.5, ha='center', color='#7C3AED', fontweight='bold')

    # MLflow -> FastAPI (Model Artifact)
    ax.annotate("", xy=(7.3, 2.2), xytext=(6.7, 2.2),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=2, mutation_scale=15, linestyle="dashed"))
    ax.text(7.0, 2.4, "Load Model (.pkl)", fontsize=8.5, ha='center', color='#475569', fontstyle='italic')

    ax.set_xlim(0, 14)
    ax.set_ylim(0.2, 8.0)
    ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

def create_cicd_pipeline_diagram(output_path):
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    
    # Title
    ax.text(7.5, 7.0, "Figure 4.4: CI/CD Pipeline Design & Stage Flow", 
            fontsize=18, fontweight='bold', ha='center', color='#0F172A', fontfamily='sans-serif')
    ax.text(7.5, 6.6, "Automated Jenkins Declarative Pipeline Execution Lifecycle", 
            fontsize=12, ha='center', color='#64748B', fontfamily='sans-serif')
    
    # 5 Stages
    stages = [
        ("Stage 1", "Checkout Code", ["• Pull commit from SCM", "• Verify git branch: main", "• Validate workspace"], "#3B82F6"),
        ("Stage 2", "Dependencies", ["• Python 3.13 verification", "• python -m pip install", "• Check requirements.txt"], "#0EA5E9"),
        ("Stage 3", "Automated Tests", ["• pytest tests/ -v", "• 5/5 Unit & API Tests", "• Generate results.xml"], "#10B981"),
        ("Stage 4", "Train & MLflow", ["• Run src/train.py", "• Log 5 metrics to MLflow", "• Save churn_model.pkl"], "#F59E0B"),
        ("Stage 5", "Deployment", ["• Model existence smoke test", "• Health-check validation", "• Ready for API serving"], "#8B5CF6")
    ]
    
    start_x = 0.6
    card_w = 2.4
    card_h = 3.6
    gap = 0.55
    y_pos = 2.4
    
    for i, (stage_num, title, bullets, color) in enumerate(stages):
        x = start_x + i * (card_w + gap)
        
        # Shadow
        shadow = patches.FancyBboxPatch((x+0.05, y_pos-0.05), card_w, card_h, boxstyle="round,pad=0.1,rounding_size=0.12",
                                        facecolor="#CBD5E1", edgecolor="none", alpha=0.35)
        ax.add_patch(shadow)
        
        # Main card
        card = patches.FancyBboxPatch((x, y_pos), card_w, card_h, boxstyle="round,pad=0.1,rounding_size=0.12",
                                      facecolor="#FFFFFF", edgecolor=color, linewidth=2.5)
        ax.add_patch(card)
        
        # Badge circle for stage number
        badge = patches.Circle((x + 0.35, y_pos + card_h - 0.4), 0.25, facecolor=color, edgecolor="none")
        ax.add_patch(badge)
        ax.text(x + 0.35, y_pos + card_h - 0.4, str(i+1), color="#FFFFFF", fontsize=11, fontweight='bold', ha='center', va='center')
        
        # Stage Label & Title
        ax.text(x + 0.72, y_pos + card_h - 0.32, stage_num.upper(), fontsize=9, fontweight='bold', color=color)
        ax.text(x + 0.15, y_pos + card_h - 0.85, title, fontsize=12, fontweight='bold', color="#1E293B")
        
        # Underline
        ax.plot([x + 0.15, x + card_w - 0.15], [y_pos + card_h - 1.05, y_pos + card_h - 1.05], color=color, linewidth=1.5, alpha=0.4)
        
        # Bullet points
        curr_y = y_pos + card_h - 1.45
        for b in bullets:
            ax.text(x + 0.15, curr_y, b, fontsize=9.5, color="#334155")
            curr_y -= 0.42
            
        # Status icon at bottom of card
        status_box = patches.FancyBboxPatch((x + 0.2, y_pos + 0.2), card_w - 0.4, 0.45, boxstyle="round,pad=0.05,rounding_size=0.08",
                                            facecolor="#DCFCE7", edgecolor="#22C55E", linewidth=1.2)
        ax.add_patch(status_box)
        ax.text(x + card_w/2, y_pos + 0.42, "✓ PASSED (SUCCESS)", fontsize=8.5, fontweight='bold', color="#15803D", ha='center')
        
        # Connector arrow to next stage
        if i < len(stages) - 1:
            arrow_x = x + card_w + 0.05
            ax.annotate("", xy=(arrow_x + gap - 0.1, y_pos + card_h/2), xytext=(arrow_x, y_pos + card_h/2),
                        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=3.0, mutation_scale=18))

    # Bottom Banner: Post Actions & Feedback
    post_box = patches.FancyBboxPatch((0.6, 0.7), 13.8, 1.2, boxstyle="round,pad=0.1,rounding_size=0.12",
                                      facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.8)
    ax.add_patch(post_box)
    ax.text(0.9, 1.45, "POST-EXECUTION GATES & AUDIT FEEDBACK", fontsize=11, fontweight='bold', color="#1E293B")
    ax.text(0.9, 1.15, "• Automated JUnit XML Archival (results.xml recorded in Jenkins)", fontsize=9.5, color="#475569")
    ax.text(0.9, 0.88, "• Build Result Status Notification (Pipeline SUCCESS: 100% stage pass rate in 2m 14s)", fontsize=9.5, color="#475569")
    ax.text(7.6, 1.15, "• MLflow Tracking Linkage (Run UUID & Parameter Metadata Recorded)", fontsize=9.5, color="#475569")
    ax.text(7.6, 0.88, "• Jira Defect Verification & Ticket State Transition (CHURN-5 Resolved)", fontsize=9.5, color="#475569")

    ax.set_xlim(0, 15)
    ax.set_ylim(0.2, 7.5)
    ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"CI/CD pipeline diagram successfully saved to {output_path}")

if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(proj_dir, "figures")
    
    arch_path = os.path.join(fig_dir, "architecture_diagram.png")
    cicd_path = os.path.join(fig_dir, "cicd_pipeline_diagram.png")
    
    create_architecture_diagram(arch_path)
    create_cicd_pipeline_diagram(cicd_path)
    
    # Also copy to artifact dir so they can be viewed directly
    artifact_dir = r"C:\Users\kirth\.gemini\antigravity\brain\443b5fff-f552-4f89-ac79-3cc476c13fc4"
    if os.path.exists(artifact_dir):
        shutil.copy(arch_path, os.path.join(artifact_dir, "architecture_diagram.png"))
        shutil.copy(cicd_path, os.path.join(artifact_dir, "cicd_pipeline_diagram.png"))
        print("Copied diagrams to artifact directory.")
