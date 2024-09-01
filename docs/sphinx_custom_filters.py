def basename(fullname):
    """Extract the basename from a full class path."""
    return fullname.split(".")[-1]


def setup(app):
    app.builder.templates.environment.filters["basename"] = basename
