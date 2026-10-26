import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { MoodSelector } from "./MoodSelector";

const moods = Array.from({ length: 32 }, (_, index) => ({ id: `mood_${index + 1}`, label: `Mood ${index + 1}`, group: "Test group", description: `Description ${index + 1}`, synonyms: [], classifier_weights: { joy: 1 }, sentiment: "positive" as const }));

it("keeps all 32 moods searchable and permits optional deselection", () => {
  const onChange = vi.fn();
  render(<MoodSelector moods={moods} selected="mood_1" onChange={onChange} label="Add a feeling" />);
  fireEvent.change(screen.getByRole("searchbox", { name: "Find another mood" }), { target: { value: "Mood 32" } });
  expect(screen.getByRole("button", { name: /Mood 32/ })).toBeInTheDocument();
  expect(screen.queryByText("Test group")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Clear selected mood" }));
  expect(onChange).toHaveBeenCalledWith("");
});
