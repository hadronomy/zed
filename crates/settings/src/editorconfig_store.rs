use anyhow::{Context as _, Result};
use collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use ec4rs::{ConfigParser, PropertiesSource, Section};
use fs::Fs;
use futures::StreamExt;
use gpui::{AppContext as _, Context, EventEmitter, Task};
use paths::EDITORCONFIG_NAME;
use smallvec::SmallVec;
use std::{path::Path, str::FromStr, sync::Arc};
use util::{ResultExt as _, rel_path::RelPath};

use crate::{
    InvalidSettingsError, LocalSettingsPath, SettingsStore, WorktreeId, watch_config_file,
};

const PARSE_CACHE_MAX_ENTRIES: usize = 32;

pub type EditorconfigProperties = ec4rs::Properties;

#[derive(Clone)]
pub struct Editorconfig {
    pub is_root: bool,
    pub sections: SmallVec<[Section; 5]>,
}

impl FromStr for Editorconfig {
    type Err = anyhow::Error;

    fn from_str(contents: &str) -> Result<Self, Self::Err> {
        let parser = ConfigParser::new_buffered(contents.as_bytes())
            .context("creating editorconfig parser")?;
        let is_root = parser.is_root;
        let sections = parser
            .collect::<Result<SmallVec<_>, _>>()
            .context("parsing editorconfig sections")?;
        Ok(Self { is_root, sections })
    }
}

#[derive(Clone, Debug)]
pub enum EditorconfigEvent {
    ExternalConfigChanged {
        path: LocalSettingsPath,
        content: Option<String>,
        affected_worktree_ids: Vec<WorktreeId>,
    },
}

impl EventEmitter<EditorconfigEvent> for EditorconfigStore {}

#[derive(Default)]
pub struct EditorconfigStore {
    external_configs: BTreeMap<Arc<Path>, (Arc<str>, Option<Arc<Editorconfig>>)>,
    worktree_state: BTreeMap<WorktreeId, EditorconfigWorktreeState>,
    parse_cache: HashMap<Arc<str>, std::result::Result<Arc<Editorconfig>, String>>,
    parse_tasks: HashMap<(WorktreeId, LocalSettingsPath), Task<()>>,
    local_external_config_watchers: BTreeMap<Arc<Path>, Task<()>>,
    local_external_config_discovery_tasks: BTreeMap<WorktreeId, Task<()>>,
}

#[derive(Default)]
struct EditorconfigWorktreeState {
    internal_configs: BTreeMap<Arc<RelPath>, (Arc<str>, Option<Arc<Editorconfig>>)>,
    external_config_paths: BTreeSet<Arc<Path>>,
}

impl EditorconfigStore {
    pub(crate) fn set_configs(
        &mut self,
        worktree_id: WorktreeId,
        path: LocalSettingsPath,
        content: Option<&str>,
        cx: &mut Context<Self>,
    ) -> std::result::Result<(), InvalidSettingsError> {
        match (&path, content) {
            (LocalSettingsPath::InWorktree(rel_path), None) => {
                self.parse_tasks.remove(&(worktree_id, path.clone()));
                if let Some(state) = self.worktree_state.get_mut(&worktree_id) {
                    state.internal_configs.remove(rel_path);
                }
            }
            (LocalSettingsPath::OutsideWorktree(abs_path), None) => {
                self.parse_tasks.remove(&(worktree_id, path.clone()));
                if let Some(state) = self.worktree_state.get_mut(&worktree_id) {
                    state.external_config_paths.remove(abs_path);
                }
                let still_in_use = self
                    .worktree_state
                    .values()
                    .any(|state| state.external_config_paths.contains(abs_path));
                if !still_in_use {
                    self.external_configs.remove(abs_path);
                    self.local_external_config_watchers.remove(abs_path);
                }
            }
            (LocalSettingsPath::InWorktree(rel_path), Some(content)) => {
                let unchanged = self
                    .worktree_state
                    .get(&worktree_id)
                    .and_then(|state| state.internal_configs.get(rel_path))
                    .is_some_and(|entry| &*entry.0 == content);
                if !unchanged {
                    let content: Arc<str> = Arc::from(content);
                    if let Some(cached) = self.parse_cache.get(&content) {
                        let cached = cached.clone();
                        self.worktree_state
                            .entry(worktree_id)
                            .or_default()
                            .internal_configs
                            .insert(rel_path.clone(), (content, cached.as_ref().ok().cloned()));
                        if let Err(message) = cached {
                            return Err(InvalidSettingsError::Editorconfig {
                                message,
                                path: LocalSettingsPath::InWorktree(
                                    rel_path
                                        .join(RelPath::from_unix_str(EDITORCONFIG_NAME).unwrap())
                                        .into(),
                                ),
                            });
                        }
                    } else {
                        self.worktree_state
                            .entry(worktree_id)
                            .or_default()
                            .internal_configs
                            .insert(rel_path.clone(), (content.clone(), None));
                        self.spawn_parse(worktree_id, path.clone(), content, cx);
                    }
                }
            }
            (LocalSettingsPath::OutsideWorktree(abs_path), Some(content)) => {
                let state = self.worktree_state.entry(worktree_id).or_default();
                state.external_config_paths.insert(abs_path.clone());
                let unchanged = self
                    .external_configs
                    .get(abs_path)
                    .is_some_and(|entry| &*entry.0 == content);
                if !unchanged {
                    let content: Arc<str> = Arc::from(content);
                    if let Some(cached) = self.parse_cache.get(&content) {
                        let cached = cached.clone();
                        self.external_configs
                            .insert(abs_path.clone(), (content, cached.as_ref().ok().cloned()));
                        if let Err(message) = cached {
                            return Err(InvalidSettingsError::Editorconfig {
                                message,
                                path: LocalSettingsPath::OutsideWorktree(
                                    abs_path.join(EDITORCONFIG_NAME).into(),
                                ),
                            });
                        }
                    } else {
                        self.external_configs
                            .insert(abs_path.clone(), (content.clone(), None));
                        self.spawn_parse(worktree_id, path.clone(), content, cx);
                    }
                }
            }
        }
        Ok(())
    }

    fn spawn_parse(
        &mut self,
        worktree_id: WorktreeId,
        path: LocalSettingsPath,
        content: Arc<str>,
        cx: &mut Context<Self>,
    ) {
        let background_parse = cx.background_spawn({
            let content = content.clone();
            async move { content.parse::<Editorconfig>().map_err(|e| e.to_string()) }
        });
        let key = (worktree_id, path);
        let task = cx.spawn({
            let key = key.clone();
            async move |this, cx| {
                let parse_result = background_parse.await.map(Arc::new);
                this.update(cx, |this, cx| {
                    if this.parse_cache.len() >= PARSE_CACHE_MAX_ENTRIES {
                        this.parse_cache.clear();
                    }
                    this.parse_cache
                        .insert(content.clone(), parse_result.clone());

                    let (worktree_id, path) = &key;
                    if let Err(message) = &parse_result {
                        log::error!("Failed to parse .editorconfig in {path:?}: {message}");
                    }
                    let entry = match path {
                        LocalSettingsPath::InWorktree(rel_path) => this
                            .worktree_state
                            .get_mut(worktree_id)
                            .and_then(|state| state.internal_configs.get_mut(rel_path)),
                        LocalSettingsPath::OutsideWorktree(abs_path) => {
                            this.external_configs.get_mut(abs_path)
                        }
                    };
                    if let Some(entry) = entry.filter(|entry| entry.0 == content) {
                        entry.1 = parse_result.ok();
                        if cx.has_global::<SettingsStore>() {
                            cx.global_mut::<SettingsStore>();
                        }
                    }
                })
                .ok();
            }
        });
        self.parse_tasks.insert(key, task);
    }

    pub(crate) fn remove_for_worktree(&mut self, root_id: WorktreeId) {
        self.local_external_config_discovery_tasks.remove(&root_id);
        self.parse_tasks.retain(|(id, _), _| id != &root_id);
        let Some(removed) = self.worktree_state.remove(&root_id) else {
            return;
        };
        let paths_in_use: HashSet<_> = self
            .worktree_state
            .values()
            .flat_map(|w| w.external_config_paths.iter())
            .collect();
        for path in removed.external_config_paths.iter() {
            if !paths_in_use.contains(path) {
                self.external_configs.remove(path);
                self.local_external_config_watchers.remove(path);
            }
        }
    }

    fn internal_configs(
        &self,
        root_id: WorktreeId,
    ) -> impl '_ + Iterator<Item = (&RelPath, &str, Option<&Editorconfig>)> {
        self.worktree_state
            .get(&root_id)
            .into_iter()
            .flat_map(|state| {
                state
                    .internal_configs
                    .iter()
                    .map(|(path, data)| (path.as_ref(), data.0.as_ref(), data.1.as_deref()))
            })
    }

    fn external_configs(
        &self,
        worktree_id: WorktreeId,
    ) -> impl '_ + Iterator<Item = (&Path, &str, Option<&Editorconfig>)> {
        self.worktree_state
            .get(&worktree_id)
            .into_iter()
            .flat_map(|state| {
                state.external_config_paths.iter().filter_map(|path| {
                    self.external_configs
                        .get(path)
                        .map(|entry| (path.as_ref(), entry.0.as_ref(), entry.1.as_deref()))
                })
            })
    }

    pub fn local_editorconfig_settings(
        &self,
        worktree_id: WorktreeId,
    ) -> impl '_ + Iterator<Item = (LocalSettingsPath, &str, Option<&Editorconfig>)> {
        let external = self
            .external_configs(worktree_id)
            .map(|(path, content, parsed)| {
                (
                    LocalSettingsPath::OutsideWorktree(path.into()),
                    content,
                    parsed,
                )
            });
        let internal = self
            .internal_configs(worktree_id)
            .map(|(path, content, parsed)| {
                (LocalSettingsPath::InWorktree(path.into()), content, parsed)
            });
        external.chain(internal)
    }

    pub fn discover_local_external_configs_chain(
        &mut self,
        worktree_id: WorktreeId,
        worktree_path: Arc<Path>,
        fs: Arc<dyn Fs>,
        cx: &mut Context<Self>,
    ) {
        // We should only have one discovery task per worktree.
        if self
            .local_external_config_discovery_tasks
            .contains_key(&worktree_id)
        {
            return;
        }

        let task = cx.spawn({
            let fs = fs.clone();
            async move |this, cx| {
                let discovered_paths = {
                    let mut paths = Vec::new();
                    let mut current = worktree_path.parent().map(|p| p.to_path_buf());
                    while let Some(dir) = current {
                        let dir_path: Arc<Path> = Arc::from(dir.as_path());
                        let path = dir.join(EDITORCONFIG_NAME);
                        if fs.load(&path).await.is_ok() {
                            paths.push(dir_path);
                        }
                        current = dir.parent().map(|p| p.to_path_buf());
                    }
                    paths
                };

                this.update(cx, |this, cx| {
                    for dir_path in discovered_paths {
                        // We insert it here so that watchers can send events to appropriate worktrees.
                        // external_config_paths gets populated again in set_configs.
                        this.worktree_state
                            .entry(worktree_id)
                            .or_default()
                            .external_config_paths
                            .insert(dir_path.clone());
                        match this.local_external_config_watchers.entry(dir_path.clone()) {
                            std::collections::btree_map::Entry::Occupied(_) => {
                                if let Some(existing_config) = this.external_configs.get(&dir_path)
                                {
                                    cx.emit(EditorconfigEvent::ExternalConfigChanged {
                                        path: LocalSettingsPath::OutsideWorktree(dir_path),
                                        content: Some(existing_config.0.to_string()),
                                        affected_worktree_ids: vec![worktree_id],
                                    });
                                } else {
                                    log::error!("Watcher exists for {dir_path:?} but no config found in external_configs");
                                }
                            }
                            std::collections::btree_map::Entry::Vacant(entry) => {
                                let watcher =
                                    Self::watch_local_external_config(fs.clone(), dir_path, cx);
                                entry.insert(watcher);
                            }
                        }
                    }
                })
                .ok();
            }
        });

        self.local_external_config_discovery_tasks
            .insert(worktree_id, task);
    }

    fn watch_local_external_config(
        fs: Arc<dyn Fs>,
        dir_path: Arc<Path>,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        let config_path = dir_path.join(EDITORCONFIG_NAME);
        let (mut config_rx, watcher_task) =
            watch_config_file(cx.background_executor(), fs, config_path);

        cx.spawn(async move |this, cx| {
            let _watcher_task = watcher_task;
            while let Some(content) = config_rx.next().await {
                let content = Some(content).filter(|c| !c.is_empty());
                let dir_path = dir_path.clone();
                this.update(cx, |this, cx| {
                    let affected_worktree_ids: Vec<WorktreeId> = this
                        .worktree_state
                        .iter()
                        .filter_map(|(id, state)| {
                            state
                                .external_config_paths
                                .contains(&dir_path)
                                .then_some(*id)
                        })
                        .collect();

                    cx.emit(EditorconfigEvent::ExternalConfigChanged {
                        path: LocalSettingsPath::OutsideWorktree(dir_path),
                        content,
                        affected_worktree_ids,
                    });
                })
                .ok();
            }
        })
    }

    pub fn properties(
        &self,
        for_worktree: WorktreeId,
        for_path: &RelPath,
    ) -> Option<EditorconfigProperties> {
        let mut properties = EditorconfigProperties::new();
        let state = self.worktree_state.get(&for_worktree);
        let internal_root_config_is_root = state
            .and_then(|state| state.internal_configs.get(RelPath::empty()))
            .and_then(|data| data.1.as_ref())
            .is_some_and(|ec| ec.is_root);

        let std_path = for_path.as_std_path();

        if !internal_root_config_is_root {
            for (_, _, parsed_editorconfig) in self.external_configs(for_worktree) {
                if let Some(parsed_editorconfig) = parsed_editorconfig {
                    if parsed_editorconfig.is_root {
                        properties = EditorconfigProperties::new();
                    }
                    for section in &parsed_editorconfig.sections {
                        section.apply_to(&mut properties, std_path).log_err()?;
                    }
                }
            }
        }

        if let Some(state) = state {
            let mut internal_configs: SmallVec<[&Editorconfig; 8]> = SmallVec::new();

            for ancestor in for_path.ancestors() {
                if let Some((_, parsed)) = state.internal_configs.get(ancestor) {
                    let config = parsed.as_deref()?;
                    internal_configs.push(config);
                    if config.is_root {
                        break;
                    }
                }
            }

            for config in internal_configs.into_iter().rev() {
                if config.is_root {
                    properties = EditorconfigProperties::new();
                }
                for section in &config.sections {
                    section.apply_to(&mut properties, std_path).log_err()?;
                }
            }
        }

        properties.use_fallbacks();
        Some(properties)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl EditorconfigStore {
    pub fn test_state(&self) -> (Vec<WorktreeId>, Vec<Arc<Path>>, Vec<Arc<Path>>) {
        let worktree_ids: Vec<_> = self.worktree_state.keys().copied().collect();
        let external_paths: Vec<_> = self.external_configs.keys().cloned().collect();
        let watcher_paths: Vec<_> = self
            .local_external_config_watchers
            .keys()
            .cloned()
            .collect();
        (worktree_ids, external_paths, watcher_paths)
    }

    pub fn external_config_paths_for_worktree(&self, worktree_id: WorktreeId) -> Vec<Arc<Path>> {
        self.worktree_state
            .get(&worktree_id)
            .map(|state| state.external_config_paths.iter().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ec4rs::property::IndentSize;
    use gpui::TestAppContext;

    #[gpui::test]
    async fn test_parses_in_background(cx: &mut TestAppContext) {
        let store = cx.new(|_| EditorconfigStore::default());
        let worktree_id = WorktreeId::from_usize(1);
        let root = LocalSettingsPath::InWorktree(Arc::from(RelPath::empty()));
        let file_path = RelPath::from_unix_str("src/main.rs").unwrap();
        let content = "root = true\n\n[*]\nindent_size = 4\n";

        store.update(cx, |store, cx| {
            store
                .set_configs(worktree_id, root.clone(), Some(content), cx)
                .unwrap();
            assert!(store.properties(worktree_id, file_path).is_none());
        });
        cx.run_until_parked();

        let properties = store
            .read_with(cx, |store, _| store.properties(worktree_id, file_path))
            .unwrap();
        assert_eq!(
            properties.get::<IndentSize>().ok(),
            Some(IndentSize::Value(4))
        );
    }

    #[gpui::test]
    async fn test_parse_cache_survives_removal(cx: &mut TestAppContext) {
        let store = cx.new(|_| EditorconfigStore::default());
        let worktree_id = WorktreeId::from_usize(1);
        let root = LocalSettingsPath::InWorktree(Arc::from(RelPath::empty()));
        let file_path = RelPath::from_unix_str("src/main.rs").unwrap();
        let content = "root = true\n\n[*]\nindent_size = 4\n";

        store.update(cx, |store, cx| {
            store
                .set_configs(worktree_id, root.clone(), Some(content), cx)
                .unwrap();
        });
        cx.run_until_parked();

        store.update(cx, |store, cx| {
            store
                .set_configs(worktree_id, root.clone(), None, cx)
                .unwrap();
            let properties = store.properties(worktree_id, file_path).unwrap();
            assert_eq!(properties.get::<IndentSize>().ok(), None);

            store
                .set_configs(worktree_id, root.clone(), Some(content), cx)
                .unwrap();
            let properties = store.properties(worktree_id, file_path).unwrap();
            assert_eq!(
                properties.get::<IndentSize>().ok(),
                Some(IndentSize::Value(4))
            );
        });
    }

    #[gpui::test]
    async fn test_invalid_config_error_is_cached(cx: &mut TestAppContext) {
        let store = cx.new(|_| EditorconfigStore::default());
        let worktree_id = WorktreeId::from_usize(1);
        let root = LocalSettingsPath::InWorktree(Arc::from(RelPath::empty()));
        let file_path = RelPath::from_unix_str("src/main.rs").unwrap();
        let content = "[]\nindent_size = 4\n";

        store.update(cx, |store, cx| {
            store
                .set_configs(worktree_id, root.clone(), Some(content), cx)
                .unwrap();
        });
        cx.run_until_parked();

        store.update(cx, |store, cx| {
            assert!(store.properties(worktree_id, file_path).is_none());

            store
                .set_configs(worktree_id, root.clone(), None, cx)
                .unwrap();
            let error = store
                .set_configs(worktree_id, root.clone(), Some(content), cx)
                .unwrap_err();
            assert!(matches!(error, InvalidSettingsError::Editorconfig { .. }));
            assert!(store.properties(worktree_id, file_path).is_none());
        });
    }
}
