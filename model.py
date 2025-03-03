from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Link(BaseModel):
    id: int
    source_type: Literal["function", "class", "module", "enum", "docs"]
    source_id: int
    target_type: Literal["function", "class", "module", "enum", "docs"]
    target_id: int
    type: Optional[str] = None


class Parameter(BaseModel):
    name: str = Field(
        ..., description="Parameter name, can be for class constructor or function"
    )
    type: Optional[str] = None
    docs_id: int
    description: Optional[str] = None
    default_value: Optional[str] = None
    is_optional: Optional[bool] = None


class Example(BaseModel):
    type: str = Field(
        ...,
        description="type is to describe the usecase of that code snippet , task which we can accomplish using this code",
    )
    code: str = Field(
        ..., description="example of the code snippet using particular function, class, etc"
    )
    description: Optional[str] = None


class Function(BaseModel):
    id: int
    module_id: Optional[int] = None
    docs_id: int
    class_id: Optional[int] = None
    name: str = Field(
        ...,
        description="Function/method name it can be associated with class or it can be pure function from module",
    )
    type: str = "function"
    description: Optional[str] = None
    signature: Optional[str] = None
    parameters: Optional[List[Parameter]] = None
    examples: Optional[List[Example]] = None
    url: Optional[str] = None


class Class(BaseModel):
    id: int
    module_id: int
    docs_id: int
    type: str = "class"
    name: str = Field(..., description="Class we that we export from a module")
    description: Optional[str] = None
    url: Optional[str] = None


class Enum(BaseModel):
    id: int
    module_id: int
    docs_id: int
    type: str = "enum"
    name: str = Field(..., description="Enum name")
    description: Optional[str] = None
    url: Optional[str] = None


class EnumValue(BaseModel):
    id: int
    enum_id: int
    docs_id: int
    name: str = Field(..., description="Enum value name")
    value: Optional[str] = None
    description: Optional[str] = None


class Module(BaseModel):
    id: int
    docs_id: int
    type: str = "module"
    name: str = Field(..., description="Name of the module")
    description: Optional[str] = None
    url: Optional[str] = None


class Docs(BaseModel):
    id: int
    url: Optional[str] = None
    type: str = "doc"
    description: str = Field(
        ...,
        description="Includes the additional information helps to clearly understand this doc which we are scrapping",
    )

class DocsLink(BaseModel):
    url: str