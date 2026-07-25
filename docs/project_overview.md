# SecureSight

SecureSight is a Security Incident Classification and SOC Analytics platform that utilizes machine learning to identify security incidents, suggest remediation measures, rank dangers, and deliver interactive cybersecurity analysis with the help of the Microsoft GUIDE database.

## Introduction to the problem

Enterprise security operations centers face large, continuously changing queues of incidents that must be triaged, investigated, and remediated. This workload creates several related but distinct decision problems.

First, analysts must determine whether an incident represents a true positive, benign positive, or false positive. Second, they must determine which remediation actions are appropriate for the affected assets. Finally, because not every incident can be investigated immediately, analysts must decide which incidents in the current queue deserve attention first.

Fully autonomous remediation requires exceptionally high confidence because an incorrect action may disable legitimate users, devices, applications, or other business-critical assets. This challenge has motivated guided response systems that provide context-rich recommendations while preserving analyst oversight.

Extended Detection and Response (XDR) products are well positioned to support these tasks because they consolidate telemetry across endpoints, identities, email, cloud environments, network activity, and other security products. This broad visibility enables models to reconstruct incident context, predict incident outcomes, recommend remediation actions, and prioritize incidents according to their relative investigation urgency.

GUIDE supports research across these complementary SOC workflows by providing real-world labels for incident triage, remediation recommendation, and queue-relative prioritization.

## Data Source and Info

The project will utilize the Microsoft GUIDE (Generalized Incident Understanding Dataset for Enterprises) dataset.
[link to the data](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction/data?select=GUIDE_Test.csv)

### The Dataset contains
 - Over 13 million cybersecurity records
 - Around 1.6 million security alerts
 - Around 1 million security incidents
 - Data from more than 6,100 organizations
 - Over 9,100 detector identifiers
 - 33 different entity types
 - 441 MITRE ATT&CK techniques

*The project will use relevant incident-level features such as:*

 - Incident ID
 - Alert Severity
 - Detector ID
 - MITRE ATT&CK Technique
 - Organization ID
 - Evidence Type
 - Alert Category
 - Detection Source
 - Incident Label (Target Variable)

## Notes on data

 - The data set seems to be fresh (updated 14 hours ago) by microsoft and 4 different collaborators
 - Objective (of security operations centres SOC) is to identify, prioritize,investigate, and respond to cybersecurity incidents
 - SOC tasks - Incident triage, remediation action recommendation, incident polarization
 - Incident (Cybersecurity): An incident is any event that compromises or threatens the confidentiality, integrity, or availability of an organization's systems, data, or networks. 
 - Incident Triage: Incident triage is the process of analyzing, prioritizing, and classifying security incidents based on their severity, impact, and urgency so they can be handled effectively. 
 - Remediation Action: A remediation action is the step taken to eliminate a security threat, fix the vulnerability, and restore affected systems to a secure state. 
 - Incident Prioritization: Incident prioritization is the process of ranking security incidents based on their severity, business impact, and urgency to determine the order in which they should be addressed. 
 - Well the data seems to be having more than 13 million data points across 33 entity types, including 1.6 million alerts(my assumption is that there must be a column directing if a row is on alert or not) and there are 1 million out of those 1.6 are with customer provided triage labels ( i think if the customer label a data point that’s a strong signal of it being an ‘incident’)
 - Out of this 1.6 lakhs 26,000 include remediation-action labels. 
 - There is another data that contains expert-derived priority rankings for 9,980 incidents across 499 organization queues, enabling researchers to evaluate which incidents should receive analyst attention first rather than only predicting their eventual triage outcomes. 
 - Problems and introductions: So enterprise security operations centers face large, continuously changing queues of incidents that must be triaged,investigated, and remediated. This workload creates several related but distinct decision problems.
 - First, analysts must determine whether an incident represents a true positive, benign positive, or false positive. Second, they must determine which remediation actions are appropriate for the affected assets. Finally, because not every incident can be investigated immediately, analysts must decide which incidents in the current queue deserve attention first. 
 - Incident - Prioritization Extension -> OrgId, IncidentId, Rank - These labels capture queue-relative investigation priority, not triage outcome. It is not necessarily more likely to be a true positive than every lower-ranked incident.
 - Benign Positive- An alert is triggered, but the activity is legitimate or expected. The detection worked, but the activity is harmless. 
 - Difference between false positive and benign positive 
 - False Positive
 - The alert itself is wrong.
 - Nothing matching the detection actually occurred.
 - Benign Positive
 - The alert is technically correct.
 - The detected activity really happened, but it was legitimate or expected, not an attack.
 - The recommended primary metric is macro-F1, accompanied by per-class precision and recall
 - Data includes approximately 26,000 incidents with customer-provided remediation-action labels at both granular and aggregate levels. 
 - These labels support evaluation of guided response systems that recommend actions for affected assets. Recommended evaluation should report macro-F1 together with precision and recall for each remediation-action category. 
 - Incident Prioritization: For each organization a method ranks all eligible incidents. Precision @ K computed by comparing the method’s top-K incidents from the same queue. Results should be normalized across organisations so that organizations with larger queues do not dominate the evaluation.
 - Precision@K:
      - Precision@K is a metric used to evaluate ranking systems. It measures how many of the top K items predicted by a model are actually relevant.
      - Formula
         - Precision@K=Number of relevant items in the top K predictions/K
      - Where:
         - K = the number of top-ranked items being evaluated (e.g., 5, 10, 20).
         - Relevant items = the items that the ground truth (experts) considers important.
 - Recommended reporting: Precision@5 , 10 , 20
 - Customised with our project our objective- Concluding ones:
 - Each row in the dataset represents an alert generated by Microsoft's XDR system. An alert is not necessarily a real cyber attack, it is simply something suspicious enough for the detection system to flag.Every alert is therefore a candidate security event, not necessarily a confirmed incident.
 - We can call each alert as positives meaning the system or their system has considered it to be a positive case for alert, so such alert if it was actually an harmful incident is a case of true positive, now an alert if it was not of any harm can be called a false positive, and if an alert is what can be considered harmful but the action was expected or intentional this case is called benign positive.
 - There are approximately 26,000 incidents for which security analysts have assigned remediation action labels. Using these labeled examples, we train a model to predict the most appropriate remediation action for a new incident based on its features. 
 - Incident Prioritization- So In a crowd of alerts we need to rank a few say K incidents which in order needs immediate response a method ranks all eligible incidents. Precision@K is computed by comparing the method's top-K incidents against the expert-selected top-K incidents from the same queue. Results should be macro-averaged across organization queues so that organizations with larger queues do not dominate the evaluation. 
 - For the incident prioritization task, the training data does not provide priority labels. Therefore, we cannot directly train a model on expert rankings. Instead, we use the incident features to develop a method that assigns a priority score to each incident. We then compare the resulting ranking with the expert rankings provided only for the test set to evaluate how well our method performs.
