#!/usr/bin/env python3
"""Fetch SRA metadata for PRJNA842344"""
import urllib.request, json, xml.etree.ElementTree as ET

# Step 1: get SRA IDs
url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra&term=PRJNA842344&retmax=100&retmode=json'
with urllib.request.urlopen(url) as r:
    data = json.loads(r.read())
    ids = data['esearchresult']['idlist']
    print(f"Found {len(ids)} SRA entries\n")

# Step 2: fetch metadata
id_str = ','.join(ids)
url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id={id_str}&retmode=xml'
with urllib.request.urlopen(url) as r:
    xml_data = r.read().decode()

root = ET.fromstring(xml_data)

header = "{:<15} {:<30} {:<15} {:<15} {:<15} {:<50}".format(
    'Run', 'Library', 'Strategy', 'Source', 'Platform', 'Sample/Treatment')
print(header)
print('-' * 140)

for pkg in root.findall('.//EXPERIMENT_PACKAGE'):
    exp = pkg.find('EXPERIMENT')
    run = pkg.find('.//RUN')
    sample = pkg.find('SAMPLE')

    run_acc = run.get('accession', '?') if run is not None else '?'

    # Library info
    lib = exp.find('.//LIBRARY_DESCRIPTOR') if exp is not None else None
    lib_name = '?'
    strategy = '?'
    source = '?'
    if lib is not None:
        ln = lib.find('LIBRARY_NAME')
        lib_name = ln.text if ln is not None and ln.text else '?'
        st = lib.find('LIBRARY_STRATEGY')
        strategy = st.text if st is not None and st.text else '?'
        sr = lib.find('LIBRARY_SOURCE')
        source = sr.text if sr is not None and sr.text else '?'

    # Platform
    platform = '?'
    pe = exp.find('.//PLATFORM') if exp is not None else None
    if pe is not None:
        for child in pe:
            model = child.find('INSTRUMENT_MODEL')
            platform = model.text if model is not None else child.tag
            break

    # Sample title
    title = '?'
    if sample is not None:
        t = sample.find('.//TITLE')
        if t is not None and t.text:
            title = t.text

    # Sample attributes
    attrs = {}
    if sample is not None:
        for attr in sample.findall('.//SAMPLE_ATTRIBUTE'):
            tag_e = attr.find('TAG')
            val_e = attr.find('VALUE')
            tag = tag_e.text if tag_e is not None else ''
            val = val_e.text if val_e is not None else ''
            attrs[tag] = val

    cell_line = attrs.get('cell_line', attrs.get('cell line', ''))
    treatment = attrs.get('treatment', attrs.get('Treatment', ''))
    source_name = attrs.get('source_name', '')

    info = f"CL={cell_line}"
    if treatment:
        info += f" | Tx={treatment}"
    if source_name:
        info += f" | {source_name}"

    line = "{:<15} {:<30} {:<15} {:<15} {:<15} {}".format(
        run_acc, lib_name[:28], strategy, source, platform[:13], info)
    print(line)
