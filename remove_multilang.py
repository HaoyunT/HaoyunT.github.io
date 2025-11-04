#!/usr/bin/env python3
import re

# Read the HTML file
with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove data-en and data-zh attributes while keeping the English content
# Pattern: data-en="English text" data-zh="Chinese text">English text<
pattern = r' data-en="([^"]*)" data-zh="[^"]*"'
content = re.sub(pattern, '', content)

# Write the cleaned content back
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully removed all multilingual attributes!")