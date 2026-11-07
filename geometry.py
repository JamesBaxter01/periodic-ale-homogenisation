"""User-supplied periodic square-channel RVE, in unit-cell coordinates."""


def square_geo(channel_width=0.33):
    """Unit cube with three connected square channels and periodic faces.

    No fine local mesh-size restrictions override the requested coarse maxh.
    Reference porosity is 3*w**2 - 2*w**3.
    """
    from netgen.occ import Box, Pnt, X, Y, Z, Glue, Translation, IdentificationType
    if not 0 < channel_width < 1:
        raise ValueError("channel_width must lie in (0, 1)")
    a, b = (1-channel_width)/2, (1+channel_width)/2
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    fluid = (Box(Pnt(0, a, a), Pnt(1, b, b))
             + Box(Pnt(a, 0, a), Pnt(b, 1, b))
             + Box(Pnt(a, a, 0), Pnt(b, b, 1)))
    solid = box - fluid
    for shape, name in ((solid, "solid"), (fluid, "fluid")):
        shape.mat(name)
        shape.faces.name = "interface"
        for axis, label in ((X, "x"), (Y, "y"), (Z, "z")):
            shape.faces.Min(axis).name = f"{label}min_{name}"
            shape.faces.Max(axis).name = f"{label}max_{name}"
    result = Glue([solid, fluid])
    for label, shift in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        for name in ("solid", "fluid"):
            result.faces[f"{label}min_{name}"].Identify(
                result.faces[f"{label}max_{name}"], f"{label}-periodic",
                type=IdentificationType.PERIODIC, trafo=Translation(shift))
    return result


def square_column(cells=10, channel_width=0.33):
    """A column of ``cells`` square_geo cells stacked along z, in unit-cell coordinates.

    Each cell is the square_geo cell (same solid and channels). The column is
    periodic in x and y only. Its bottom and top faces are named
    zmin_solid/zmin_fluid and zmax_solid/zmax_fluid, solid--fluid faces
    "interface", lateral faces {x,y}{min,max}_{solid,fluid}, and the faces
    between cells (inside the column) cells_solid and cells_fluid: these join
    two cells of the same phase and are not solid--fluid interface.
    """
    from netgen.occ import Box, Pnt, Glue, Translation, IdentificationType
    if not 0 < channel_width < 1:
        raise ValueError("channel_width must lie in (0, 1)")
    if int(cells) != cells or cells < 1:
        raise ValueError("cells must be a positive integer")
    a, b = (1-channel_width)/2, (1+channel_width)/2
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    fluid = (Box(Pnt(0, a, a), Pnt(1, b, b))
             + Box(Pnt(a, 0, a), Pnt(b, 1, b))
             + Box(Pnt(a, a, 0), Pnt(b, b, 1)))
    solid = box - fluid
    parts = []
    for i in range(cells):
        for template, name in ((solid, "solid"), (fluid, "fluid")):
            shape = template.Move((0, 0, i))
            shape.mat(name)
            shape.faces.name = "interface"
            for face in shape.faces:
                centre = tuple(face.center)
                for axis, label, extent in ((0, "x", 1), (1, "y", 1), (2, "z", cells)):
                    if abs(centre[axis]) < 1e-8:
                        face.name = f"{label}min_{name}"
                    if abs(centre[axis] - extent) < 1e-8:
                        face.name = f"{label}max_{name}"
                if abs(centre[2] - round(centre[2])) < 1e-8 and 0 < round(centre[2]) < cells:
                    face.name = f"cells_{name}"  # shared with the next cell's same phase
            parts.append(shape)
    result = Glue(parts)
    for label, shift in (("x", (1, 0, 0)), ("y", (0, 1, 0))):
        for name in ("solid", "fluid"):
            sources = list(result.faces[f"{label}min_{name}"])
            targets = list(result.faces[f"{label}max_{name}"])
            for index, source in enumerate(sources):
                centre = tuple(c + s for c, s in zip(source.center, shift))
                matches = [face for face in targets if sum((c - t)**2 for c, t in zip(face.center, centre)) < 1e-16]
                if len(matches) != 1:
                    raise ValueError("ambiguous periodic face pairing")
                source.Identify(matches[0], f"{label}-periodic-{name}-{index}",
                                type=IdentificationType.PERIODIC, trafo=Translation(shift))
    return result
