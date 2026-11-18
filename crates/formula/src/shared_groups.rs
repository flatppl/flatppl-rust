use crate::{Error, Recipe, ResolvedPrior, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The coefficient interval owned by a term in a joint innovation block.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(crate) struct Coefficients {
    pub offset: usize,
    pub width: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Block {
    pub group: String,
    pub members: Vec<(String, usize)>,
    pub width: usize,
    pub prior: ResolvedPrior,
}

pub(crate) fn plan(recipe: &Recipe) -> Result<Vec<Block>> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut ids = HashMap::new();
    for (name, predictor) in recipe.predictors() {
        for (index, (group, term)) in predictor
            .groups
            .iter()
            .zip(&predictor.formula.groups)
            .enumerate()
        {
            let Some(id) = &term.id else { continue };
            let block_number = ids.entry(id).or_insert(blocks.len());
            if *block_number == blocks.len() {
                blocks.push(Block {
                    group: group.column.clone(),
                    members: Vec::new(),
                    width: 0,
                    prior: ResolvedPrior::new("lkj(1)"),
                });
            }
            let block = &mut blocks[*block_number];
            if block.group != group.column {
                return Err(Error(format!(
                    "group ID `{id}` must use the same grouping factor in every term"
                )));
            }
            block.members.push((name.to_owned(), index));
            block.width += group.coefficients.len();
        }
    }
    blocks.retain(|block| block.width > 1);
    Ok(blocks)
}
