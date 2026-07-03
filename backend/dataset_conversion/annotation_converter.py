from __future__ import annotations

from backend.dataset_conversion.exceptions import OntologyAdapterError
from backend.dataset_conversion.interfaces import (
    AnnotationConverterInterface,
    OntologyAdapterInterface,
)
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CoordinateSystem,
    GeometryType,
    SourceAnnotation,
)
from backend.dataset_conversion.ontology_adapter import OntologyAdapter


class AnnotationConverter(AnnotationConverterInterface):
    def __init__(
        self,
        ontology_adapter: OntologyAdapterInterface | None = None,
    ) -> None:
        self._ontology_adapter = ontology_adapter or OntologyAdapter()

    async def convert_annotation(
        self,
        source: SourceAnnotation,
        canonical_label: str,
    ) -> CanonicalAnnotation:
        x, y, w, h = await self._to_canonical_geometry(source)

        canonical_name = canonical_label.replace(".", "_").replace("-", "_").title()

        return CanonicalAnnotation(
            image_id=str(source.image_id),
            canonical_label=canonical_label,
            canonical_name=canonical_name,
            geometry_type=self._output_geometry_type(source.geometry_type),
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=source.confidence if source.confidence is not None else 1.0,
            source_annotation_id=source.id,
            source_label=source.category_name,
            metadata={
                "source_category_id": str(source.category_id),
                **source.metadata,
            },
        )

    async def convert_batch(
        self,
        sources: list[SourceAnnotation],
        label_map: dict[str, str],
    ) -> list[CanonicalAnnotation]:
        results: list[CanonicalAnnotation] = []
        for src in sources:
            canonical = label_map.get(src.category_name)
            if canonical is None:
                raise OntologyAdapterError(f"No ontology mapping for label '{src.category_name}'")
            results.append(await self.convert_annotation(src, canonical))
        return results

    async def _to_canonical_geometry(
        self,
        source: SourceAnnotation,
    ) -> tuple[float, float, float, float]:
        coords = source.coordinates
        if source.geometry_type == GeometryType.BBOX:
            if source.coordinate_system == CoordinateSystem.NORMALIZED:
                if source.image_width and source.image_height:
                    return (
                        coords[0] * source.image_width,
                        coords[1] * source.image_height,
                        coords[2] * source.image_width,
                        coords[3] * source.image_height,
                    )
                return (coords[0], coords[1], coords[2], coords[3])
            return (coords[0], coords[1], coords[2], coords[3])
        elif source.geometry_type == GeometryType.NORMALIZED:
            if source.image_width and source.image_height:
                return (
                    coords[0] * source.image_width,
                    coords[1] * source.image_height,
                    coords[2] * source.image_width,
                    coords[3] * source.image_height,
                )
            return (coords[0], coords[1], coords[2], coords[3])
        elif source.geometry_type == GeometryType.POLYGON:
            xs = coords[0::2]
            ys = coords[1::2]
            xmin = min(xs)
            ymin = min(ys)
            xmax = max(xs)
            ymax = max(ys)
            return (xmin, ymin, xmax - xmin, ymax - ymin)
        else:
            return (coords[0], coords[1], coords[2], coords[3])

    def _output_geometry_type(self, source_type: GeometryType) -> GeometryType:
        if source_type in (GeometryType.BBOX, GeometryType.NORMALIZED):
            return GeometryType.BBOX
        return source_type
