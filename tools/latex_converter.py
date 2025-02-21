from typing import Dict, List, Optional
import re
from datetime import datetime

class LaTeXResumeConverter:
    def __init__(self):
        self.latex_special_chars = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
    
    def escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not isinstance(text, str):
            return str(text)
        for char, replacement in self.latex_special_chars.items():
            text = text.replace(char, replacement)
        return text

    def generate_document_header(self) -> List[str]:
        """Generate the LaTeX document header."""
        return [
            "\\documentclass[letterpaper,11pt]{article}",
            "\\usepackage{latexsym}",
            "\\usepackage[empty]{fullpage}",
            "\\usepackage{titlesec}",
            "\\usepackage{marvosym}",
            "\\usepackage[usenames,dvipsnames]{color}",
            "\\usepackage{verbatim}",
            "\\usepackage{enumitem}",
            "\\usepackage[hidelinks]{hyperref}",
            "\\usepackage{fancyhdr}",
            "\\usepackage[english]{babel}",
            "\\usepackage{tabularx}",
            "\\input{glyphtounicode}",
            "",
            "\\pagestyle{fancy}",
            "\\fancyhf{} % clear all header and footer fields",
            "\\fancyfoot{}",
            "\\renewcommand{\\headrulewidth}{0pt}",
            "\\renewcommand{\\footrulewidth}{0pt}",
            "",
            "% Adjust margins",
            "\\addtolength{\\oddsidemargin}{-0.5in}",
            "\\addtolength{\\evensidemargin}{-0.5in}",
            "\\addtolength{\\textwidth}{1in}",
            "\\addtolength{\\topmargin}{-.5in}",
            "\\addtolength{\\textheight}{1.0in}",
            "",
            "\\urlstyle{same}",
            "\\raggedbottom",
            "\\raggedright",
            "\\setlength{\\tabcolsep}{0in}",
            "",
            "% Custom commands",
            self._get_custom_commands(),
            ""
        ]

    def _get_custom_commands(self) -> str:
        """Define custom LaTeX commands."""
        return """
\\newcommand{\\resumeItem}[1]{
  \\item\\small{
    {#1 \\vspace{-2pt}}
  }
}

\\newcommand{\\resumeSubheading}[4]{
  \\vspace{-2pt}\\item
    \\begin{tabular*}{0.97\\textwidth}[t]{l@{\\extracolsep{\\fill}}r}
      \\textbf{#1} & #2 \\\\
      \\textit{\\small#3} & \\textit{\\small #4} \\\\
    \\end{tabular*}\\vspace{-7pt}
}

\\newcommand{\\resumeSubSubheading}[2]{
    \\item
    \\begin{tabular*}{0.97\\textwidth}{l@{\\extracolsep{\\fill}}r}
      \\textit{\\small#1} & \\textit{\\small #2} \\\\
    \\end{tabular*}\\vspace{-7pt}
}"""

    def format_personal_info(self, info: Dict) -> List[str]:
        """Format personal information section."""
        return [
            "\\begin{center}",
            f"    \\textbf{{\\Huge \\scshape {self.escape_latex(info['name'])}}} \\\\ \\vspace{{1pt}}",
            "    \\small " + 
            f"{self.escape_latex(info['phone'])} $|$ " +
            f"\\underline{{{self.escape_latex(info['email'])}}} $|$ " +
            f"\\href{{{info['linkedin']}}}" +
            f"{{\\underline{{linkedin.com/in/{info['linkedin'].split('/')[-1]}}}}} $|$ " +
            f"\\href{{{info['github']}}}" +
            f"{{\\underline{{github.com/{info['github'].split('/')[-1]}}}}}",
            "\\end{center}",
            ""
        ]

    def format_skills(self, skills: Dict) -> List[str]:
        """Format skills section."""
        return [
            "\\section{Technical Skills}",
            "\\begin{itemize}[leftmargin=0.15in, label={}]",
            "    \\small{\\item{",
            *[f"     \\textbf{{{category}}}{{: {', '.join(items)}}}" + 
              (" \\\\" if i < len(skills) - 1 else "")
              for i, (category, items) in enumerate(skills.items())],
            "    }}",
            "\\end{itemize}",
            ""
        ]

    def format_experience(self, experiences: List[Dict]) -> List[str]:
        """Format experience section."""
        lines = [
            "\\section{Experience}",
            "\\resumeSubHeadingListStart"
        ]
        
        for exp in experiences:
            lines.extend([
                "  \\resumeSubheading",
                f"    {{{self.escape_latex(exp['title'])}}}{{{exp['start_date']} -- {exp['end_date']}}}",
                f"    {{{self.escape_latex(exp['company'])}}}{{{exp['location']}}}",
                "  \\resumeItemListStart"
            ])
            
            for responsibility in exp['responsibilities']:
                lines.append(f"    \\resumeItem{{{self.escape_latex(responsibility)}}}")
            
            lines.extend([
                "  \\resumeItemListEnd",
                ""
            ])
        
        lines.append("\\resumeSubHeadingListEnd")
        return lines

    def convert_json_to_latex(self, json_data: Dict) -> str:
        """Convert JSON resume data to LaTeX format."""
        latex_content = []
        
        # Add document header
        latex_content.extend(self.generate_document_header())
        
        # Begin document
        latex_content.extend([
            "\\begin{document}",
            ""
        ])
        
        # Add sections
        if 'personal_info' in json_data:
            latex_content.extend(self.format_personal_info(json_data['personal_info']))
        
        if 'skills' in json_data:
            latex_content.extend(self.format_skills(json_data['skills']))
            
        if 'experience' in json_data:
            latex_content.extend(self.format_experience(json_data['experience']))
        
        # Close document
        latex_content.append("\\end{document}")
        
        return "\n".join(latex_content)