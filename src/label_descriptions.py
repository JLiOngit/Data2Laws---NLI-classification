import pandas as pd

LABEL_DESCRIPTIONS = pd.DataFrame([
    ["Style", "readability, code layout, indentation issues"],
    ["Naming", "naming variables, methods, classes"],
    ["Questioning", "questions to the author, requests clarification"],
    ["Response", "agreements with others, reviewer appointments"],
    ["Convention", "software development process discussion"],
    ["Testing", "requests tests to verify functionality"],
    ["Design", "architecture and code design, program structure"],
    ["Refactoring", "logical structure, object creation"],
    ["Functionality", "identification of code defects"],
    ["Roadmap", "further development of the program"],
    ["Optimization", "code optimization, parallelism"],
    ["Error", "exception and error handling problems"],
    ["Documentation", "documentation or source code comments"],
    ["Support", "compatibility with other systems"],
    ["Input/Output", "GUI input/output, pop-up windows"],
    ["Other", "comments without semantic context"],
], columns=['label', 'label_description'])


GROUP_DESCRIPTIONS = pd.DataFrame([
    ["CodeStyle", "formatting, naming, readability, structure"],
    ["Discussion", "opening discussion, ask for rationale"],
    ["Development", "code correctness, bugs, performance, security"],
    ["User", "user experience, usability, UI behavior"],
    ["Other", "documentation, project structure, general remarks"]
], columns=['group', 'group_description'])