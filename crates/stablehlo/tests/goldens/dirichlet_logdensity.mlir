module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<3xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.compare LE, %1, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%arg0 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.constant dense<8.0> : tensor<3xf32>
    %11 = stablehlo.compare GE, %arg0, %10 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %12 = stablehlo.select %11, %arg0, %10 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %15 = stablehlo.divide %14, %12 : tensor<3xf32>
    %16 = stablehlo.multiply %15, %15 : tensor<3xf32>
    %17 = stablehlo.constant dense<-0.00165322962780713> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %19 = stablehlo.multiply %18, %16 : tensor<3xf32>
    %20 = stablehlo.constant dense<0.000837308034031215> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %22 = stablehlo.add %19, %21 : tensor<3xf32>
    %23 = stablehlo.multiply %22, %16 : tensor<3xf32>
    %24 = stablehlo.constant dense<-0.00059520293135187> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %26 = stablehlo.add %23, %25 : tensor<3xf32>
    %27 = stablehlo.multiply %26, %16 : tensor<3xf32>
    %28 = stablehlo.constant dense<0.00079365066682539> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %30 = stablehlo.add %27, %29 : tensor<3xf32>
    %31 = stablehlo.multiply %30, %16 : tensor<3xf32>
    %32 = stablehlo.constant dense<-0.00277777777760991> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %34 = stablehlo.add %31, %33 : tensor<3xf32>
    %35 = stablehlo.multiply %34, %16 : tensor<3xf32>
    %36 = stablehlo.constant dense<0.0833333333333333> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %38 = stablehlo.add %35, %37 : tensor<3xf32>
    %39 = stablehlo.multiply %38, %15 : tensor<3xf32>
    %40 = chlo.lgamma %arg0 : tensor<3xf32> -> tensor<3xf32>
    %41 = stablehlo.constant dense<0.5> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %43 = stablehlo.subtract %arg0, %42 : tensor<3xf32>
    %44 = stablehlo.log %arg0 : tensor<3xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<3xf32>
    %46 = stablehlo.subtract %40, %45 : tensor<3xf32>
    %47 = stablehlo.add %46, %arg0 : tensor<3xf32>
    %48 = stablehlo.constant dense<0.9189385332046727> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %50 = stablehlo.subtract %47, %49 : tensor<3xf32>
    %51 = stablehlo.select %11, %39, %50 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %52 = stablehlo.reduce(%51 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %53 = stablehlo.constant dense<8.0> : tensor<f32>
    %54 = stablehlo.compare GE, %9, %53 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %55 = stablehlo.select %54, %9, %53 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %56 = stablehlo.divide %13, %55 : tensor<f32>
    %57 = stablehlo.multiply %56, %56 : tensor<f32>
    %58 = stablehlo.multiply %17, %57 : tensor<f32>
    %59 = stablehlo.add %58, %20 : tensor<f32>
    %60 = stablehlo.multiply %59, %57 : tensor<f32>
    %61 = stablehlo.add %60, %24 : tensor<f32>
    %62 = stablehlo.multiply %61, %57 : tensor<f32>
    %63 = stablehlo.add %62, %28 : tensor<f32>
    %64 = stablehlo.multiply %63, %57 : tensor<f32>
    %65 = stablehlo.add %64, %32 : tensor<f32>
    %66 = stablehlo.multiply %65, %57 : tensor<f32>
    %67 = stablehlo.add %66, %36 : tensor<f32>
    %68 = stablehlo.multiply %67, %56 : tensor<f32>
    %69 = chlo.lgamma %9 : tensor<f32> -> tensor<f32>
    %70 = stablehlo.subtract %9, %41 : tensor<f32>
    %71 = stablehlo.log %9 : tensor<f32>
    %72 = stablehlo.multiply %70, %71 : tensor<f32>
    %73 = stablehlo.subtract %69, %72 : tensor<f32>
    %74 = stablehlo.add %73, %9 : tensor<f32>
    %75 = stablehlo.subtract %74, %48 : tensor<f32>
    %76 = stablehlo.select %54, %68, %75 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %77 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %78 = stablehlo.divide %arg0, %77 : tensor<3xf32>
    %79 = stablehlo.log %78 : tensor<3xf32>
    %80 = stablehlo.multiply %43, %79 : tensor<3xf32>
    %81 = stablehlo.reduce(%80 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %82 = stablehlo.constant dense<0.5> : tensor<3xf32>
    %83 = stablehlo.reduce(%82 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %84 = stablehlo.subtract %83, %41 : tensor<f32>
    %85 = stablehlo.constant dense<1.8378770664093453> : tensor<f32>
    %86 = stablehlo.subtract %71, %85 : tensor<f32>
    %87 = stablehlo.multiply %84, %86 : tensor<f32>
    %88 = stablehlo.subtract %52, %76 : tensor<f32>
    %89 = stablehlo.add %88, %81 : tensor<f32>
    %90 = stablehlo.subtract %89, %87 : tensor<f32>
    %91 = stablehlo.constant dense<1.0> : tensor<3xf32>
    %92 = stablehlo.subtract %arg0, %91 : tensor<3xf32>
    %93 = stablehlo.constant dense<"0x1002CEBFC81B9ABF187231BF"> : tensor<3xf32>
    %94 = stablehlo.multiply %92, %93 : tensor<3xf32>
    %95 = stablehlo.reduce(%94 init: %8) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %96 = stablehlo.subtract %95, %90 : tensor<f32>
    return %96, %6 : tensor<f32>, tensor<i32>
  }
}
