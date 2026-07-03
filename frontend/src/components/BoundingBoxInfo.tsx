import type { DetectedObject } from "../types";

interface BoundingBoxInfoProps {
  objects: DetectedObject[];
}

export default function BoundingBoxInfo({ objects }: BoundingBoxInfoProps) {
  return (
    <div className="mt-3">
      <h4 className="mb-2 text-[10px] font-medium uppercase tracking-wider text-gray-500">
        Spatial Coordinates
      </h4>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-dss-border text-left text-gray-500">
              <th className="pb-1 pr-2 font-medium">Object</th>
              <th className="pb-1 pr-2 font-medium">X</th>
              <th className="pb-1 pr-2 font-medium">Y</th>
              <th className="pb-1 pr-2 font-medium">Width</th>
              <th className="pb-1 pr-2 font-medium">Height</th>
              <th className="pb-1 font-medium">Area</th>
            </tr>
          </thead>
          <tbody className="text-gray-400">
            {objects.slice(0, 10).map((obj) => (
              <tr key={obj.id} className="border-b border-dss-border/20">
                <td className="py-1 pr-2 font-mono text-[10px] text-gray-500">
                  {obj.id.slice(0, 8)}
                </td>
                <td className="py-1 pr-2 font-mono">
                  {obj.bounding_box.x.toFixed(1)}
                </td>
                <td className="py-1 pr-2 font-mono">
                  {obj.bounding_box.y.toFixed(1)}
                </td>
                <td className="py-1 pr-2 font-mono">
                  {obj.bounding_box.width.toFixed(1)}
                </td>
                <td className="py-1 pr-2 font-mono">
                  {obj.bounding_box.height.toFixed(1)}
                </td>
                <td className="py-1 font-mono text-dss-muted">
                  {(
                    obj.bounding_box.width * obj.bounding_box.height
                  ).toFixed(0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
