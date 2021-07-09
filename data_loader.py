import problog


def load(filename):
    parser = problog.parser.PrologParser(problog.program.ExtendedPrologFactory())
    parsed = parser.parseFile(filename)
    return parsed


def parse_query(query):
    parser = problog.parser.PrologParser(problog.program.ExtendedPrologFactory())
    parsed = parser.parseString(query)
    return parsed
